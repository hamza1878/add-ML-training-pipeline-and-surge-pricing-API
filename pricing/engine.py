from __future__ import annotations

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
# Cherche cleaned_data.csv puis le CSV brut en fallback
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

    ALL = [ECONOMY, STANDARD, COMFORT, FIRST_CLASS, VAN, MINI_BUS]

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
        mapping = {
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
        }
        return mapping.get(s, cls.COMFORT)


# ══════════════════════════════════════════════════════════════════
# DATACLASS RÉSULTAT
# ══════════════════════════════════════════════════════════════════

@dataclass
class PriceResult:
    base_fare:          float
    distance_cost:      float
    duration_cost:      float
    raw_price:          float
    surge_multiplier:   float
    final_price:        float
    currency:           str   = "TND"
    min_applied:        bool  = False
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
    ml_used:            bool  = False
    ml_surge_xgb:       Optional[float] = None
    ml_surge_lgbm:      Optional[float] = None
    ml_confidence:      Optional[float] = None
    source:             str   = "Règles métier"
    labels:             dict  = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
# CALCUL PAR RÈGLES MÉTIER
# ══════════════════════════════════════════════════════════════════

def compute_price_rules(
    distance_km:  float,
    duration_min: float,
    row:          dict,
    car_type:     str = CarType.COMFORT,
) -> PriceResult:
    car_type = CarType.normalize(car_type)

    dist_cost = round(distance_km  * RATE_PER_KM,  2)
    dur_cost  = round(duration_min * RATE_PER_MIN,  2)
    raw       = round(BASE_FARE + dist_cost + dur_cost, 2)

    m_traffic = MULT_TRAFFIC.get(int(row.get("trafic_niveau",  1)), 1.0)
    m_weather = MULT_WEATHER.get(int(row.get("weather_code",   1)), 1.0)
    m_demand  = MULT_DEMAND.get(str(row.get("demande", "normal")), 1.0)
    m_night   = MULT_NIGHT if row.get("is_night") else 1.0
    m_car     = MULT_CAR.get(car_type, 1.0)
    m_friday  = MULT_FRIDAY_JUMUAH if row.get("is_friday_slot") else 1.0
    m_zone    = MULT_ZONE.get(str(row.get("zone_type", "intérieure")), 1.0)

    ram_key = "none"
    if row.get("is_aid_el_fitr"):
        ram_key = "none"
    elif row.get("is_ramadan_slot"):
        p = str(row.get("periode", "")).lower()
        if   "iftar"  in p: ram_key = "ramadan_iftar"
        elif "taraw"  in p: ram_key = "ramadan_tarawih"
        elif "suhoor" in p: ram_key = "ramadan_suhoor"
        else:               ram_key = "ramadan_iftar"
    elif row.get("is_ramadan_last_week"):
        ram_key = "ramadan_last_week"
    m_ramadan = MULT_RAMADAN.get(ram_key, 1.0)

    beach_key = (
        str(row.get("beach_peak_reason", "none"))
        if row.get("is_beach_hour") else "none"
    )
    m_beach = MULT_BEACH.get(beach_key, 1.0)

    special   = str(row.get("special_event", "none"))
    m_special = MULT_SPECIAL_EVENT.get(special, 1.0)

    surge = round(
        m_traffic * m_weather * m_demand * m_night
        * m_car * m_friday * m_ramadan * m_beach * m_zone * m_special,
        4,
    )
    final = round(raw * surge, 2)
    min_applied = False
    if final < MIN_FARE:
        final, min_applied = MIN_FARE, True

    labels = {
        "traffic":       f"×{m_traffic} (niveau {row.get('trafic_niveau', 1)})",
        "weather":       f"×{m_weather} (code {row.get('weather_code', 1)} — {row.get('weather_label','?')})",
        "demand":        f"×{m_demand}  ({row.get('demande', 'normal')})",
        "night":         f"×{m_night}   (is_night={row.get('is_night', 0)})",
        "car":           f"×{m_car}     ({car_type} — {CarType.LABELS.get(car_type, car_type)})",
        "friday":        f"×{m_friday}  (is_friday={row.get('is_friday_slot', 0)})",
        "ramadan":       f"×{m_ramadan} ({ram_key})",
        "beach":         f"×{m_beach}   ({beach_key})",
        "zone":          f"×{m_zone}    ({row.get('zone_type', 'intérieure')})",
        "special_event": f"×{m_special} ({special})",
    }

    return PriceResult(
        base_fare=BASE_FARE, distance_cost=dist_cost, duration_cost=dur_cost,
        raw_price=raw, surge_multiplier=surge, final_price=final,
        min_applied=min_applied,
        mult_traffic=m_traffic, mult_weather=m_weather, mult_demand=m_demand,
        mult_night=m_night, mult_car=m_car, mult_friday=m_friday,
        mult_ramadan=m_ramadan, mult_beach=m_beach, mult_zone=m_zone,
        mult_special_event=m_special,
        ml_used=False, source="Règles métier pures", labels=labels,
    )


# ══════════════════════════════════════════════════════════════════
# CALCUL VIA SURGE ML
# ══════════════════════════════════════════════════════════════════

def compute_price_ml(
    distance_km:  float,
    duration_min: float,
    row:          dict,
    ml_surge:     float,
    car_type:     str = CarType.COMFORT,
) -> PriceResult:
    car_type  = CarType.normalize(car_type)
    dist_cost = round(distance_km  * RATE_PER_KM,  2)
    dur_cost  = round(duration_min * RATE_PER_MIN,  2)
    raw       = round(BASE_FARE + dist_cost + dur_cost, 2)
    m_car     = MULT_CAR.get(car_type, 1.0)
    special   = str(row.get("special_event", "none"))
    m_special = MULT_SPECIAL_EVENT.get(special, 1.0)

    surge_total = round(ml_surge * m_car * m_special, 4)
    final       = round(raw * surge_total, 2)
    min_applied = False
    if final < MIN_FARE:
        final, min_applied = MIN_FARE, True

    rules = compute_price_rules(distance_km, duration_min, row, car_type)
    rules.surge_multiplier  = surge_total
    rules.final_price       = final
    rules.min_applied       = min_applied
    rules.ml_used           = True
    rules.source            = (
        f"ML ensemble (XGB×0.55 + LGBM×0.45)  "
        f"surge_raw={ml_surge:.4f}  car=×{m_car}  special=×{m_special} ({special})"
    )
    rules.labels["ml"] = (
        f"surge_ml={ml_surge:.4f} × car={m_car} × special={m_special}"
    )
    return rules


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
    dataset_csv:       str   = "",   # chemin optionnel vers le CSV pour geo_lookup
) -> dict:
    """
    Pipeline complet Moviroo.

    NOUVEAU :
      • geo_lookup : cherche les coordonnées dans le dataset (rayon 20 km)
        et enrichit automatiquement zone_type, has_beach, population,
        intensite_ville si un point est trouvé.
      • La météo (weather_code, temperature_2m, windspeed_10m, precipitation)
        est transmise au ML dans le row — le modèle la voit maintenant.
    """
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
    geo_meta_origin = None
    geo_meta_dest   = None

    if lookup and lookup.loaded:
        geo_meta_origin = lookup.find_nearest(lat_origin, lon_origin)
        geo_meta_dest   = lookup.find_nearest(lat_dest,   lon_dest)

    # Enrichissement depuis le dataset si trouvé (origine prioritaire)
    geo_meta = geo_meta_origin or geo_meta_dest
    if geo_meta:
        # On surcharge uniquement si les valeurs passées sont les défauts
        zone_type       = geo_meta.get("zone_type",       zone_type)
        has_beach       = int(geo_meta.get("has_beach",   has_beach))
        population      = int(geo_meta.get("population",  population))
        intensite_ville = int(geo_meta.get("intensite_ville", intensite_ville))
        ville_label     = geo_meta.get("ville", "?")
        print(f"     ✅ Origine : {ville_label} / {zone_type} | beach={has_beach}")
    else:
        print("     ℹ️  Coordonnées hors dataset — valeurs fournies conservées")

    if geo_meta_dest and geo_meta_dest != geo_meta_origin:
        print(f"     ✅ Destination : {geo_meta_dest.get('ville','?')} / {geo_meta_dest.get('zone_type','?')}")

    # ── 3. Météo ──────────────────────────────────────────────────
    _step(3, "Météo Open-Meteo")
    weather = fetch_weather(lat_origin, lon_origin, booking_dt)
    estimated_tag = " [ESTIMÉE]" if weather.get("weather_estimated") else ""
    print(
        f"     {weather['weather_label'].capitalize()}{estimated_tag} | "
        f"{weather['temperature_2m']}°C | "
        f"Vent {weather['windspeed_10m']} km/h | "
        f"is_night={weather['is_night']}"
    )

    # ── 4. Flags temporels ────────────────────────────────────────
    _step(4, "Flags temporels & culturels")
    time_flags  = compute_time_flags(booking_dt)
    beach_flags = compute_beach_flags(has_beach, booking_dt)
    print(f"     Période : {time_flags['periode']}  |  Saison : {time_flags['season']}")

    se = time_flags.get("special_event", "none")
    if se != "none":
        print(f"     🎉 Événement : {se}  (×{MULT_SPECIAL_EVENT.get(se, 1.0)})")
    if time_flags.get("is_ramadan_last_week"):
        print(f"     🌙 Dernière semaine Ramadan (×{MULT_RAMADAN['ramadan_last_week']})")
    if beach_flags["is_beach_hour"]:
        print(
            f"     🏖️  Beach surge ×{beach_flags['beach_surge_value']} "
            f"({beach_flags['beach_peak_reason']})"
        )

    # ── Assemblage du contexte ─────────────────────────────────
    row = {
        "zone_type":           zone_type,
        "has_beach":           has_beach,
        "population":          population,
        "intensite_ville":     intensite_ville,
        "trafic_niveau":       trafic_niveau,
        "demande":             demande,
        "indice_congestion":   indice_congestion,
        "retard_estime_min":   retard_estime_min,
        "vitesse_moy_kmh":     vitesse_moy_kmh,
        "chauffeurs_actifs":   chauffeurs_actifs,
        "car_type":            car_type,
        # Météo — transmise au ML (CORRECTIF)
        "weather_code":        weather["weather_code"],
        "weather_label":       weather["weather_label"],
        "temperature_2m":      weather["temperature_2m"],
        "windspeed_10m":       weather["windspeed_10m"],
        "precipitation":       weather.get("precipitation", 0.0),
        "is_night":            weather["is_night"],
        # Flags
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

    # ── 6. Résultat ───────────────────────────────────────────────
    output = {
        "base_fare":          result.base_fare,
        "distance_cost":      result.distance_cost,
        "duration_cost":      result.duration_cost,
        "raw_price":          result.raw_price,
        "surge_multiplier":   result.surge_multiplier,
        "final_price":        result.final_price,
        "currency":           "TND",
        "min_applied":        result.min_applied,
        "mult_traffic":       result.mult_traffic,
        "mult_weather":       result.mult_weather,
        "mult_demand":        result.mult_demand,
        "mult_night":         result.mult_night,
        "mult_car":           result.mult_car,
        "mult_friday":        result.mult_friday,
        "mult_ramadan":       result.mult_ramadan,
        "mult_beach":         result.mult_beach,
        "mult_zone":          result.mult_zone,
        "mult_special_event": result.mult_special_event,
        "ml_used":            result.ml_used,
        "ml_surge_xgb":       result.ml_surge_xgb,
        "ml_surge_lgbm":      result.ml_surge_lgbm,
        "ml_confidence":      result.ml_confidence,
        "source":             result.source,
        "labels":             result.labels,
        "distance_km":        distance_km,
        "duration_min":       duration_min,
        "car_type":           car_type,
        "car_type_label":     CarType.LABELS.get(car_type, car_type),
        "zone_type":          zone_type,
        "booking_dt":         booking_dt.isoformat(),
        "weather":            weather,
        "time_flags":         time_flags,
        "beach_flags":        beach_flags,
        "geo_origin":         geo_meta_origin,
        "geo_dest":           geo_meta_dest,
    }

    _print_result(output)
    return output


# ══════════════════════════════════════════════════════════════════
# AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def _print_result(r: dict) -> None:
    w   = 58
    ml  = r.get("ml_surge_xgb") is not None
    wth = r.get("weather", {})
    tf  = r.get("time_flags", {})
    se  = tf.get("special_event", "none")
    weather_tag = " [estimée]" if wth.get("weather_estimated") else ""
    geo = r.get("geo_origin") or {}
    ville = geo.get("ville", "")
    dist_geo = f" ({geo.get('distance_km', '?')} km du dataset)" if geo else " (hors dataset)"

    print("\n┌" + "─"*w + "┐")
    print(f"│  💳  RÉCAPITULATIF COURSE{'':<{w-26}}│")
    print("├" + "─"*w + "┤")
    print(f"│  Distance : {r['distance_km']:.2f} km  /  {r['duration_min']:.1f} min{'':<{w-37}}│")
    print(f"│  Véhicule : {r.get('car_type_label', r['car_type']):<{w-13}}│")
    if ville:
        print(f"│  Ville    : {ville}{dist_geo:<{w-13-len(ville)}}│")
    print(f"│  Zone     : {r['zone_type']:<{w-13}}│")
    print(f"│  Météo    : {wth.get('weather_label','?').capitalize()}{weather_tag} {wth.get('temperature_2m','?')}°C | code={wth.get('weather_code','?'):<{w-40}}│")
    print(f"│  Heure    : {tf.get('heure_int', '?'):02}h  Saison : {tf.get('season','?'):<{w-25}}│")
    if se != "none":
        print(f"│  🎉 Événement : {se:<{w-17}}│")
    if tf.get("is_ramadan_last_week"):
        print(f"│  🌙 Fin Ramadan (dernière semaine){'':<{w-36}}│")
    print("├" + "─"*w + "┤")
    print(f"│  Base       : {BASE_FARE:.2f} TND{'':<{w-22}}│")
    print(f"│  Distance   : {r['distance_cost']:.2f} TND{'':<{w-22}}│")
    print(f"│  Durée      : {r['duration_cost']:.2f} TND{'':<{w-22}}│")
    print(f"│  Brut       : {r['raw_price']:.2f} TND{'':<{w-22}}│")
    print("├" + "─"*w + "┤")
    print(f"│  Mult trafic  : ×{r['mult_traffic']}{'':<{w-20}}│")
    print(f"│  Mult météo   : ×{r['mult_weather']}  (weather_code={wth.get('weather_code','?')}){'':<{w-44}}│")
    if r.get("mult_special_event", 1.0) != 1.0:
        print(f"│  Mult spécial : ×{r['mult_special_event']} ({se}){'':<{w-32}}│")
    if ml:
        print(f"│  XGB          : ×{r['ml_surge_xgb']:.4f}{'':<{w-22}}│")
        print(f"│  LGBM         : ×{r['ml_surge_lgbm']:.4f}{'':<{w-22}}│")
        print(f"│  Confiance    : {r.get('ml_confidence',0)*100:.1f}%{'':<{w-19}}│")
    print(f"│  Surge total  : ×{r['surge_multiplier']:.4f}{'':<{w-22}}│")
    min_note = " ← minimum" if r.get("min_applied") else ""
    print("╞" + "═"*w + "╡")
    print(f"│  💰 PRIX FINAL : {r['final_price']:.2f} TND{min_note:<{w-26}}│")
    print("└" + "─"*w + "┘")


def _sep(title: str) -> None:
    print("\n" + "═"*60)
    print(f"  {title}")
    print("═"*60)


def _step(n: int, title: str) -> None:
    print(f"\n  [{n}] {title}")