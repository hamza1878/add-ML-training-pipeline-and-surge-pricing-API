from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime    import datetime
from typing      import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BASE_FARE, RATE_PER_KM, RATE_PER_MIN, MIN_FARE,
    MULT_TRAFFIC, MULT_WEATHER, MULT_DEMAND, MULT_NIGHT,
    MULT_CAR, MULT_FRIDAY_JUMUAH, MULT_RAMADAN, MULT_BEACH, MULT_ZONE,
    MULT_SPECIAL_EVENT,
)
from utils.routing    import get_osrm_distance
from utils.weather    import fetch_weather
from utils.flags      import compute_time_flags, compute_beach_flags
from utils.geo_lookup import DatasetLookup
from models.predictor import predictor

# ── DatasetLookup singleton ───────────────────────────────────────
_LOOKUP_CANDIDATES = ["cleaned_data.csv", "tunisia_all_cities_traffic.csv"]
_dataset_lookup: Optional[DatasetLookup] = None

def _get_lookup() -> Optional[DatasetLookup]:
    global _dataset_lookup
    if _dataset_lookup is not None:
        return _dataset_lookup
    for candidate in _LOOKUP_CANDIDATES:
        if Path(candidate).exists():
            _dataset_lookup = DatasetLookup(candidate)
            return _dataset_lookup
    print("  ⚠️  Aucun dataset trouvé pour geo_lookup — métadonnées manuelles uniquement")
    return None


# ══════════════════════════════════════════════════════════════════
# TYPES DE VÉHICULES
# ══════════════════════════════════════════════════════════════════

class CarType:
    ECONOMY     = "economy"
    STANDARD    = "standard"
    COMFORT     = "comfort"
    FIRST_CLASS = "first_class"
    VAN         = "van"
    MINI_BUS    = "mini_bus"

    ALL    = [ECONOMY, STANDARD, COMFORT, FIRST_CLASS, VAN, MINI_BUS]
    LABELS = {
        ECONOMY:     "Economy",
        STANDARD:    "Standard",
        COMFORT:     "Comfort",
        FIRST_CLASS: "First Class",
        VAN:         "Van",
        MINI_BUS:    "Mini Bus",
    }

    @classmethod
    def normalize(cls, raw: str) -> str:
        s = raw.lower().strip().replace(" ", "_").replace("-", "_")
        return {
            "economy":     cls.ECONOMY,
            "standard":    cls.STANDARD,
            "comfort":     cls.COMFORT,
            "first_class": cls.FIRST_CLASS,
            "firstclass":  cls.FIRST_CLASS,
            "first":       cls.FIRST_CLASS,
            "premium":     cls.FIRST_CLASS,
            "van":         cls.VAN,
            "mini_bus":    cls.MINI_BUS,
            "minibus":     cls.MINI_BUS,
        }.get(s, cls.COMFORT)


# ══════════════════════════════════════════════════════════════════
# SURGE SAISONNIER  (haute / basse saison Tunisie)
# ══════════════════════════════════════════════════════════════════
# Représente la hausse structurelle de la demande selon la saison.
# Distinct du weather_mult qui capte l'effet météo immédiat.
_SEASON_SURGE: dict[str, float] = {
    "été":       1.20,   # haute saison — tourisme + beach
    "printemps": 1.10,   # épaule
    "automne":   1.05,   # épaule basse
    "hiver":     1.00,   # basse saison
}


# ══════════════════════════════════════════════════════════════════
# DATACLASS RÉSULTAT
# ══════════════════════════════════════════════════════════════════

@dataclass
class PriceResult:
    base_fare:           float
    distance_cost:       float
    duration_cost:       float
    raw_price:           float
    surge_multiplier:    float
    final_price:         float
    final_price_rounded: float   # arrondi au 5 TND supérieur
    loyalty_points:      int     # prix_arrondi × 0.5, arrondi au 5 pts
    currency:            str   = "TND"
    min_applied:         bool  = False

    # ── Détail de chaque multiplicateur ──────────────────────────
    mult_traffic:       float = 1.0
    mult_weather:       float = 1.0
    mult_demand:        float = 1.0
    mult_night:         float = 1.0
    mult_car:           float = 1.0
    mult_friday:        float = 1.0
    mult_ramadan:       float = 1.0
    mult_beach:         float = 1.0
    mult_zone:          float = 1.0
    mult_special_event: float = 1.0
    mult_season:        float = 1.0

    ml_used:       bool            = False
    ml_surge_xgb:  Optional[float] = None
    ml_surge_lgbm: Optional[float] = None
    ml_confidence: Optional[float] = None

    source: str  = "Règles métier"
    labels: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
# HELPER — résolution centralisée de TOUS les multiplicateurs
# ══════════════════════════════════════════════════════════════════

def _resolve_multipliers(row: dict, car_type: str) -> dict:
    """
    Extrait chaque multiplicateur depuis `row` + config.py.
    Source unique de vérité — utilisée par les deux fonctions
    compute_price_rules ET compute_price_ml.
    """
    # Trafic
    m_traffic = MULT_TRAFFIC.get(int(row.get("trafic_niveau", 1)), 1.0)

    # Météo — on préfère weather_mult (déjà calculé par fetch_weather)
    # car il intègre la détection sirocco et la conversion WMO.
    m_weather = float(
        row.get("weather_mult")
        or MULT_WEATHER.get(int(row.get("weather_code", 1)), 1.0)
    )

    # Demande app
    m_demand = MULT_DEMAND.get(str(row.get("demande", "normal")).lower(), 1.0)

    # Nuit (MULT_NIGHT = 2.20 selon config)
    m_night = MULT_NIGHT if row.get("is_night") else 1.0

    # Véhicule
    m_car = MULT_CAR.get(CarType.normalize(car_type), 1.0)

    # Vendredi Jumu'ah
    m_friday = MULT_FRIDAY_JUMUAH if row.get("is_friday_slot") else 1.0

    # Zone géographique
    m_zone = MULT_ZONE.get(str(row.get("zone_type", "intérieure")).lower(), 1.0)

    # Ramadan / fin Ramadan
    ram_key = "none"
    if row.get("is_aid_el_fitr"):
        ram_key = "none"                   # Aïd el-Fitr géré par special_event
    elif row.get("is_ramadan_slot"):
        p = str(row.get("periode", "")).lower()
        if   "iftar"  in p: ram_key = "ramadan_iftar"
        elif "taraw"  in p: ram_key = "ramadan_tarawih"
        elif "suhoor" in p: ram_key = "ramadan_suhoor"
        else:               ram_key = "ramadan_iftar"
    elif row.get("is_ramadan_last_week"):
        ram_key = "ramadan_last_week"
    m_ramadan = MULT_RAMADAN.get(ram_key, 1.0)

    # Beach surge
    beach_key = (
        str(row.get("beach_peak_reason", "none"))
        if row.get("is_beach_hour") else "none"
    )
    m_beach = MULT_BEACH.get(beach_key, 1.0)

    # Événement spécial (Aïd el-Fitr, Aïd el-Adha, Nouvel An)
    special = str(row.get("special_event", "none")).lower()
    m_special = MULT_SPECIAL_EVENT.get(special, 1.0)

    # Saison
    season = str(row.get("season", "été")).lower()
    m_season = _SEASON_SURGE.get(season, 1.0)

    return {
        "m_traffic":  m_traffic,
        "m_weather":  m_weather,
        "m_demand":   m_demand,
        "m_night":    m_night,
        "m_car":      m_car,
        "m_friday":   m_friday,
        "m_zone":     m_zone,
        "m_ramadan":  m_ramadan,
        "m_beach":    m_beach,
        "m_special":  m_special,
        "m_season":   m_season,
        "ram_key":    ram_key,
        "beach_key":  beach_key,
        "special":    special,
        "season":     season,
    }


def _build_labels(row: dict, m: dict) -> dict:
    ct = CarType.normalize(row.get("car_type", "comfort"))
    return {
        "traffic":       f"×{m['m_traffic']}  (niveau {row.get('trafic_niveau',1)})",
        "weather":       f"×{m['m_weather']}  (code {row.get('weather_code',1)} — {row.get('weather_label','?')})",
        "demand":        f"×{m['m_demand']}  ({row.get('demande','normal')})",
        "night":         f"×{m['m_night']}  (is_night={row.get('is_night',0)})",
        "car":           f"×{m['m_car']}  ({ct} — {CarType.LABELS.get(ct,'')})",
        "friday":        f"×{m['m_friday']}  (is_friday={row.get('is_friday_slot',0)})",
        "ramadan":       f"×{m['m_ramadan']}  ({m['ram_key']})",
        "beach":         f"×{m['m_beach']}  ({m['beach_key']})",
        "zone":          f"×{m['m_zone']}  ({row.get('zone_type','intérieure')})",
        "special_event": f"×{m['m_special']}  ({m['special']})",
        "season":        f"×{m['m_season']}  ({m['season']})",
    }


def _finalize(raw: float, surge: float) -> tuple[float, float, int, bool]:
    """Prix final → arrondi 5 TND → points fidélité."""
    final = round(raw * surge, 2)
    min_applied = False
    if final < MIN_FARE:
        final, min_applied = MIN_FARE, True
    rounded = int(math.ceil(final / 5) * 5)
    loyalty = int(math.ceil(rounded * 0.5 / 5) * 5)
    return final, float(rounded), loyalty, min_applied


# ══════════════════════════════════════════════════════════════════
# CALCUL PAR RÈGLES MÉTIER PURES
# ══════════════════════════════════════════════════════════════════

def compute_price_rules(
    distance_km:  float,
    duration_min: float,
    row:          dict,
    car_type:     str = CarType.COMFORT,
) -> PriceResult:
    """
    Prix basé uniquement sur les règles métier du config.
    Multiplie : trafic × météo × demande × nuit × véhicule
               × vendredi × ramadan × beach × zone × spécial × saison
    """
    car_type = CarType.normalize(car_type)
    row = {**row, "car_type": car_type}

    dist_cost = round(distance_km  * RATE_PER_KM,  2)
    dur_cost  = round(duration_min * RATE_PER_MIN,  2)
    raw       = round(BASE_FARE + dist_cost + dur_cost, 2)

    m = _resolve_multipliers(row, car_type)

    surge = round(
        m["m_traffic"] * m["m_weather"] * m["m_demand"]
        * m["m_night"] * m["m_car"]    * m["m_friday"]
        * m["m_ramadan"] * m["m_beach"] * m["m_zone"]
        * m["m_special"] * m["m_season"],
        4,
    )

    final, rounded, loyalty, min_applied = _finalize(raw, surge)

    return PriceResult(
        base_fare=BASE_FARE, distance_cost=dist_cost, duration_cost=dur_cost,
        raw_price=raw, surge_multiplier=surge,
        final_price=final, final_price_rounded=rounded, loyalty_points=loyalty,
        min_applied=min_applied,
        mult_traffic=m["m_traffic"],     mult_weather=m["m_weather"],
        mult_demand=m["m_demand"],       mult_night=m["m_night"],
        mult_car=m["m_car"],             mult_friday=m["m_friday"],
        mult_ramadan=m["m_ramadan"],     mult_beach=m["m_beach"],
        mult_zone=m["m_zone"],           mult_special_event=m["m_special"],
        mult_season=m["m_season"],
        ml_used=False, source="Règles métier pures",
        labels=_build_labels(row, m),
    )


# ══════════════════════════════════════════════════════════════════
# CALCUL VIA SURGE ML  (ML × règles métier)
# ══════════════════════════════════════════════════════════════════

def compute_price_ml(
    distance_km:  float,
    duration_min: float,
    row:          dict,
    ml_surge:     float,
    car_type:     str = CarType.COMFORT,
) -> PriceResult:
    """
    Fusionne le surge ML avec les règles métier.

    ML a appris     : trafic, météo, demande, heure (capturés en training)
    Règles forcées  : nuit × zone × vendredi × ramadan × beach
                      × véhicule × événement_spécial × saison

    Ces règles sont appliquées multiplicativement PAR-DESSUS ml_surge
    afin d'intégrer tous les facteurs du config.
    """
    car_type = CarType.normalize(car_type)
    row = {**row, "car_type": car_type}

    dist_cost = round(distance_km  * RATE_PER_KM,  2)
    dur_cost  = round(duration_min * RATE_PER_MIN,  2)
    raw       = round(BASE_FARE + dist_cost + dur_cost, 2)

    m = _resolve_multipliers(row, car_type)

    # Règles métier non capturées par le ML
    surge_rules = round(
        m["m_night"]   * m["m_zone"]    * m["m_friday"]
        * m["m_ramadan"] * m["m_beach"] * m["m_car"]
        * m["m_special"] * m["m_season"],
        4,
    )

    surge_total = round(ml_surge * surge_rules, 4)
    final, rounded, loyalty, min_applied = _finalize(raw, surge_total)

    labels = _build_labels(row, m)
    labels["ml_raw"]    = f"surge_ML={ml_surge:.4f}"
    labels["ml_rules"]  = f"règles_métier=×{surge_rules:.4f}"
    labels["ml_fusion"] = f"total=×{ml_surge:.4f} × ×{surge_rules:.4f} = ×{surge_total:.4f}"

    return PriceResult(
        base_fare=BASE_FARE, distance_cost=dist_cost, duration_cost=dur_cost,
        raw_price=raw, surge_multiplier=surge_total,
        final_price=final, final_price_rounded=rounded, loyalty_points=loyalty,
        min_applied=min_applied,
        mult_traffic=m["m_traffic"],     mult_weather=m["m_weather"],
        mult_demand=m["m_demand"],       mult_night=m["m_night"],
        mult_car=m["m_car"],             mult_friday=m["m_friday"],
        mult_ramadan=m["m_ramadan"],     mult_beach=m["m_beach"],
        mult_zone=m["m_zone"],           mult_special_event=m["m_special"],
        mult_season=m["m_season"],
        ml_used=True,
        source=(
            f"ML (XGB×0.55 + LGBM×0.45) × règles métier | "
            f"surge_ml={ml_surge:.4f} × règles={surge_rules:.4f} = ×{surge_total:.4f}"
        ),
        labels=labels,
    )


# ══════════════════════════════════════════════════════════════════
# PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════

def calculate_trip_price(
    lat_origin:        float,
    lon_origin:        float,
    lat_dest:          float,
    lon_dest:          float,
    zone_type:         str   = "intérieure",
    has_beach:         int   = 0,
    population:        int   = 300_000,
    intensite_ville:   int   = 3,
    trafic_niveau:     int   = 1,
    demande:           str   = "normal",
    indice_congestion: int   = 30,
    retard_estime_min: int   = 5,
    vitesse_moy_kmh:   float = 40.0,
    chauffeurs_actifs: int   = 30,
    car_type:          str   = CarType.COMFORT,
    booking_dt:        datetime | None = None,
    use_ml:            bool  = True,
    dataset_csv:       str   = "",
) -> dict:
    if booking_dt is None:
        booking_dt = datetime.now()

    car_type = CarType.normalize(car_type)
    _sep("MOVIROO — Calcul prix dynamique")
    print(f"  Véhicule : {CarType.LABELS.get(car_type, car_type)}")

    # ── 1. Distance & Durée ───────────────────────────────────────
    _step(1, "Distance OSRM")
    distance_km, duration_min = get_osrm_distance(
        lat_origin, lon_origin, lat_dest, lon_dest
    )
    print(f"     {distance_km:.2f} km  |  {duration_min:.1f} min")

    # ── 2. Geo Lookup ─────────────────────────────────────────────
    _step(2, "Geo Lookup — dataset")
    lookup = _get_lookup()
    geo_meta_origin = geo_meta_dest = None

    if lookup and lookup.loaded:
        geo_meta_origin = lookup.find_nearest(lat_origin, lon_origin)
        geo_meta_dest   = lookup.find_nearest(lat_dest,   lon_dest)

    geo_meta = geo_meta_origin or geo_meta_dest
    if geo_meta:
        zone_type       = geo_meta.get("zone_type",       zone_type)
        has_beach       = int(geo_meta.get("has_beach",   has_beach))
        population      = int(geo_meta.get("population",  population))
        intensite_ville = int(geo_meta.get("intensite_ville", intensite_ville))
        print(f"     ✅ {geo_meta.get('ville','?')} / {zone_type} | beach={has_beach}")
    else:
        print("     ℹ️  Hors dataset — valeurs fournies conservées")

    if geo_meta_dest and geo_meta_dest != geo_meta_origin:
        print(f"     ✅ Dest : {geo_meta_dest.get('ville','?')} / {geo_meta_dest.get('zone_type','?')}")

    # ── 3. Météo ──────────────────────────────────────────────────
    _step(3, "Météo Open-Meteo")
    weather = fetch_weather(lat_origin, lon_origin, booking_dt)
    est = " [ESTIMÉE]" if weather.get("weather_estimated") else ""
    print(
        f"     {weather['weather_label'].capitalize()}{est} | "
        f"{weather['temperature_2m']}°C | vent {weather['windspeed_10m']} km/h | "
        f"is_night={weather['is_night']} | mult=×{weather['weather_mult']}"
    )

    # ── 4. Flags temporels ────────────────────────────────────────
    _step(4, "Flags temporels & culturels")
    time_flags  = compute_time_flags(booking_dt)
    beach_flags = compute_beach_flags(has_beach, booking_dt)

    se = time_flags.get("special_event", "none")
    print(f"     Période : {time_flags['periode']}  |  Saison : {time_flags['season']}")
    if se != "none":
        print(f"     🎉 Événement : {se}  (×{MULT_SPECIAL_EVENT.get(se,1.0)})")
    if time_flags.get("is_ramadan_last_week"):
        print(f"     🌙 Fin Ramadan (×{MULT_RAMADAN['ramadan_last_week']})")
    if beach_flags["is_beach_hour"]:
        print(f"     🏖️  Beach ×{beach_flags['beach_surge_value']} ({beach_flags['beach_peak_reason']})")

    # ── Assemblage du contexte row ────────────────────────────────
    row = {
        "zone_type":         zone_type,
        "has_beach":         has_beach,
        "population":        population,
        "intensite_ville":   intensite_ville,
        "trafic_niveau":     trafic_niveau,
        "demande":           demande,
        "indice_congestion": indice_congestion,
        "retard_estime_min": retard_estime_min,
        "vitesse_moy_kmh":   vitesse_moy_kmh,
        "chauffeurs_actifs": chauffeurs_actifs,
        "car_type":          car_type,
        # Météo — clé weather_mult transmise pour usage direct
        "weather_code":      weather["weather_code"],
        "weather_label":     weather["weather_label"],
        "weather_mult":      weather["weather_mult"],   # ← direct depuis config
        "temperature_2m":    weather["temperature_2m"],
        "windspeed_10m":     weather["windspeed_10m"],
        "precipitation":     weather.get("precipitation", 0.0),
        "is_night":          weather["is_night"],
        **time_flags,
        **beach_flags,
    }

    # ── 5. Surge ──────────────────────────────────────────────────
    _step(5, "Calcul du surge")
    result: PriceResult

    if use_ml and predictor.is_loaded:
        try:
            ml_result = predictor.predict(row)
            surge     = ml_result["surge_final"]
            result    = compute_price_ml(distance_km, duration_min, row, surge, car_type)
            result.ml_surge_xgb  = ml_result["surge_xgb"]
            result.ml_surge_lgbm = ml_result["surge_lgbm"]
            result.ml_confidence = ml_result["confidence"]
            print(
                f"     ML — XGB=×{ml_result['surge_xgb']:.3f}  "
                f"LGBM=×{ml_result['surge_lgbm']:.3f}  "
                f"Ensemble=×{surge:.3f}  "
                f"Confiance={ml_result['confidence']*100:.1f}%"
            )
        except Exception as exc:
            print(f"     ⚠️  ML erreur ({exc}) → règles métier")
            result = compute_price_rules(distance_km, duration_min, row, car_type)
    else:
        reason = "désactivé" if not use_ml else "non chargé"
        print(f"     Règles métier pures (ML {reason})")
        result = compute_price_rules(distance_km, duration_min, row, car_type)

    # ── 6. Sortie ─────────────────────────────────────────────────
    output = {
        "base_fare":            result.base_fare,
        "distance_cost":        result.distance_cost,
        "duration_cost":        result.duration_cost,
        "raw_price":            result.raw_price,
        "surge_multiplier":     result.surge_multiplier,
        "final_price":          result.final_price,
        "final_price_rounded":  result.final_price_rounded,
        "loyalty_points":       result.loyalty_points,
        "currency":             "TND",
        "min_applied":          result.min_applied,
        "mult_traffic":         result.mult_traffic,
        "mult_weather":         result.mult_weather,
        "mult_demand":          result.mult_demand,
        "mult_night":           result.mult_night,
        "mult_car":             result.mult_car,
        "mult_friday":          result.mult_friday,
        "mult_ramadan":         result.mult_ramadan,
        "mult_beach":           result.mult_beach,
        "mult_zone":            result.mult_zone,
        "mult_special_event":   result.mult_special_event,
        "mult_season":          result.mult_season,
        "ml_used":              result.ml_used,
        "ml_surge_xgb":         result.ml_surge_xgb,
        "ml_surge_lgbm":        result.ml_surge_lgbm,
        "ml_confidence":        result.ml_confidence,
        "source":               result.source,
        "labels":               result.labels,
        "distance_km":          distance_km,
        "duration_min":         duration_min,
        "car_type":             car_type,
        "car_type_label":       CarType.LABELS.get(car_type, car_type),
        "zone_type":            zone_type,
        "booking_dt":           booking_dt.isoformat(),
        "weather":              weather,
        "time_flags":           time_flags,
        "beach_flags":          beach_flags,
        "geo_origin":           geo_meta_origin,
        "geo_dest":             geo_meta_dest,
    }

    _print_result(output)
    return output


# ══════════════════════════════════════════════════════════════════
# AFFICHAGE RÉCAPITULATIF
# ══════════════════════════════════════════════════════════════════

def _print_result(r: dict) -> None:
    W   = 58
    ml  = r.get("ml_surge_xgb") is not None
    wth = r.get("weather", {})
    tf  = r.get("time_flags", {})
    bf  = r.get("beach_flags", {})
    se  = tf.get("special_event", "none")
    est = " [estimée]" if wth.get("weather_estimated") else ""
    geo = r.get("geo_origin") or {}

    def row(label: str, value: str) -> None:
        line = f"│  {label:<20}: {value}"
        print(line + " " * max(0, W + 2 - len(line) - 1) + "│")

    def hline(c="─"): print("├" + c * W + "┤")

    print("\n┌" + "─" * W + "┐")
    print(f"│{'  💳  RÉCAPITULATIF COURSE':^{W + 1}}│")
    hline()
    row("Véhicule",  r.get("car_type_label", r["car_type"]))
    if geo.get("ville"):
        row("Ville", f"{geo['ville']}  ({geo.get('distance_km','?')} km dataset)")
    row("Zone",      r["zone_type"])
    row("Saison",    tf.get("season", "?"))
    row("Heure",     f"{tf.get('heure_int','?'):02}h — {tf.get('periode','?')}")
    row("Météo",     f"{wth.get('weather_label','?').capitalize()}{est}  {wth.get('temperature_2m','?')}°C  code={wth.get('weather_code','?')}")
    if bf.get("is_beach_hour"):
        row("Beach",   f"×{bf.get('beach_surge_value',1.0)}  ({bf.get('beach_peak_reason','?')})")
    if se != "none":
        row("Événement", f"{se}  (×{r.get('mult_special_event',1.0)})")
    if tf.get("is_ramadan_last_week"):
        row("Ramadan", "dernière semaine")

    hline()
    row("Base",      f"{r['base_fare']:.2f} TND")
    row("Distance",  f"{r['distance_km']:.2f} km → {r['distance_cost']:.2f} TND")
    row("Durée",     f"{r['duration_min']:.1f} min → {r['duration_cost']:.2f} TND")
    row("Prix brut", f"{r['raw_price']:.2f} TND")

    hline()
    mults = [
        ("Trafic",   r["mult_traffic"]),
        ("Météo",    r["mult_weather"]),
        ("Demande",  r["mult_demand"]),
        ("Nuit",     r["mult_night"]),
        ("Véhicule", r["mult_car"]),
        ("Zone",     r["mult_zone"]),
        ("Vendredi", r["mult_friday"]),
        ("Ramadan",  r["mult_ramadan"]),
        ("Beach",    r["mult_beach"]),
        ("Saison",   r["mult_season"]),
        ("Spécial",  r["mult_special_event"]),
    ]
    for name, val in mults:
        tag = "  ◀" if val != 1.0 else ""
        row(f"  ×{name}", f"{val:.4f}{tag}")

    if ml:
        hline()
        row("  XGBoost",   f"×{r['ml_surge_xgb']:.4f}")
        row("  LightGBM",  f"×{r['ml_surge_lgbm']:.4f}")
        row("  Confiance", f"{r.get('ml_confidence',0)*100:.1f}%")

    hline("═")
    row("SURGE TOTAL", f"×{r['surge_multiplier']:.4f}")
    hline("═")

    exact   = r["final_price"]
    rounded = int(r["final_price_rounded"])
    pts     = r["loyalty_points"]
    min_tag = "  ← tarif minimum" if r.get("min_applied") else ""

    row("Prix exact",    f"{exact:.2f} TND{min_tag}")
    row("Prix facturé",  f"{rounded} TND  (arrondi ↑ au 5 TND)")
    row("Points fidél.", f"{pts} pts  (= {rounded} × 0.5 arrondi)")
    print("└" + "─" * W + "┘")


def _sep(title: str) -> None:
    print("\n" + "═" * 62)
    print(f"  {title}")
    print("═" * 62)


def _step(n: int, title: str) -> None:
    print(f"\n  [{n}] {title}")