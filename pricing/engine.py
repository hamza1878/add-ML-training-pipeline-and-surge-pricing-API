"""
╔══════════════════════════════════════════════════════════════════╗
║   MOVIROO — pricing/engine.py                                    ║
║   Responsabilité : calcul du prix final TND                      ║
║                                                                  ║
║   Formule :                                                      ║
║     prix = (BASE_FARE + km×0.55 + min×0.10)                     ║
║           × surge_multiplier                                     ║
║                                                                  ║
║   Fonctions publiques :                                          ║
║     compute_price_rules(distance_km, duration_min, row, car)     ║
║     compute_price_ml(distance_km, duration_min, row, surge, car) ║
║     calculate_trip_price(**kwargs)  ← pipeline complet           ║
╚══════════════════════════════════════════════════════════════════╝
"""

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
)
from utils.routing  import get_osrm_distance
from utils.weather  import fetch_weather
from utils.flags    import compute_time_flags, compute_beach_flags
from models.predictor import predictor


# ══════════════════════════════════════════════════════════════════
# DATACLASS RÉSULTAT
# ══════════════════════════════════════════════════════════════════

@dataclass
class PriceResult:
    """Résultat complet d'un calcul de prix."""

    # ── Prix ─────────────────────────────────────────────────────
    base_fare:         float
    distance_cost:     float
    duration_cost:     float
    raw_price:         float
    surge_multiplier:  float
    final_price:       float
    currency:          str  = "TND"
    min_applied:       bool = False

    # ── Multiplicateurs détaillés ─────────────────────────────────
    mult_traffic:  float = 1.0
    mult_weather:  float = 1.0
    mult_demand:   float = 1.0
    mult_night:    float = 1.0
    mult_car:      float = 1.0
    mult_friday:   float = 1.0
    mult_ramadan:  float = 1.0
    mult_beach:    float = 1.0
    mult_zone:     float = 1.0

    # ── Contexte ML ──────────────────────────────────────────────
    ml_used:       bool           = False
    ml_surge_xgb:  Optional[float] = None
    ml_surge_lgbm: Optional[float] = None
    ml_confidence: Optional[float] = None

    # ── Métadonnées ───────────────────────────────────────────────
    source:    str  = "Règles métier"
    labels:    dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
# CALCUL PAR RÈGLES MÉTIER
# ══════════════════════════════════════════════════════════════════

def compute_price_rules(
    distance_km:  float,
    duration_min: float,
    row:          dict,
    car_type:     str = "comfort",
) -> PriceResult:
    """
    Calcule le prix final TND via les règles métier pures (sans ML).
    Utilisé comme fallback ou pour l'audit/labels du dataset.

    Paramètres :
        distance_km   distance réelle (depuis OSRM ou dataset)
        duration_min  durée estimée
        row           dict contextuel (zone_type, trafic_niveau, …)
        car_type      economy | comfort | van | premium
    """
    dist_cost = round(distance_km  * RATE_PER_KM,  2)
    dur_cost  = round(duration_min * RATE_PER_MIN,  2)
    raw       = round(BASE_FARE + dist_cost + dur_cost, 2)

    # ── Multiplicateurs simples ───────────────────────────────────
    m_traffic = MULT_TRAFFIC.get(int(row.get("trafic_niveau",  1)), 1.0)
    m_weather = MULT_WEATHER.get(int(row.get("weather_code",   1)), 1.0)
    m_demand  = MULT_DEMAND.get(str(row.get("demande", "normal")), 1.0)
    m_night   = MULT_NIGHT if row.get("is_night") else 1.0
    m_car     = MULT_CAR.get(car_type, 1.0)
    m_friday  = MULT_FRIDAY_JUMUAH if row.get("is_friday_slot") else 1.0
    m_zone    = MULT_ZONE.get(str(row.get("zone_type", "intérieure")), 1.0)

    # ── Ramadan — sous-créneau ────────────────────────────────────
    ram_key = "none"
    if row.get("is_ramadan_slot"):
        p = str(row.get("periode", "")).lower()
        if   "iftar"  in p:  ram_key = "ramadan_iftar"
        elif "taraw"  in p:  ram_key = "ramadan_tarawih"
        elif "suhoor" in p:  ram_key = "ramadan_suhoor"
        else:                ram_key = "ramadan_iftar"
    m_ramadan = MULT_RAMADAN[ram_key]

    # ── Beach ─────────────────────────────────────────────────────
    beach_key = (
        str(row.get("beach_peak_reason", "none"))
        if row.get("is_beach_hour") else "none"
    )
    m_beach = MULT_BEACH.get(beach_key, 1.0)

    # ── Surge total ───────────────────────────────────────────────
    surge = round(
        m_traffic * m_weather * m_demand * m_night
        * m_car * m_friday * m_ramadan * m_beach * m_zone,
        4,
    )

    final       = round(raw * surge, 2)
    min_applied = False
    if final < MIN_FARE:
        final, min_applied = MIN_FARE, True

    labels = {
        "traffic": f"×{m_traffic} (niveau {row.get('trafic_niveau', 1)})",
        "weather": f"×{m_weather} (code {row.get('weather_code', 1)})",
        "demand":  f"×{m_demand}  ({row.get('demande', 'normal')})",
        "night":   f"×{m_night}   (is_night={row.get('is_night', 0)})",
        "car":     f"×{m_car}     ({car_type})",
        "friday":  f"×{m_friday}  (is_friday={row.get('is_friday_slot', 0)})",
        "ramadan": f"×{m_ramadan} ({ram_key})",
        "beach":   f"×{m_beach}   ({beach_key})",
        "zone":    f"×{m_zone}    ({row.get('zone_type', 'intérieure')})",
    }

    return PriceResult(
        base_fare        = BASE_FARE,
        distance_cost    = dist_cost,
        duration_cost    = dur_cost,
        raw_price        = raw,
        surge_multiplier = surge,
        final_price      = final,
        min_applied      = min_applied,
        mult_traffic     = m_traffic,
        mult_weather     = m_weather,
        mult_demand      = m_demand,
        mult_night       = m_night,
        mult_car         = m_car,
        mult_friday      = m_friday,
        mult_ramadan     = m_ramadan,
        mult_beach       = m_beach,
        mult_zone        = m_zone,
        ml_used          = False,
        source           = "Règles métier pures",
        labels           = labels,
    )


# ══════════════════════════════════════════════════════════════════
# CALCUL VIA SURGE ML
# ══════════════════════════════════════════════════════════════════

def compute_price_ml(
    distance_km:  float,
    duration_min: float,
    row:          dict,
    ml_surge:     float,
    car_type:     str = "comfort",
) -> PriceResult:
    """
    Calcule le prix final en utilisant le surge prédit par le ML.
    La base tarifaire est identique aux règles.
    Le modificateur véhicule est appliqué par-dessus le surge ML.

    Paramètres :
        ml_surge  surge prédit par l'ensemble XGB+LGBM
    """
    dist_cost = round(distance_km  * RATE_PER_KM,  2)
    dur_cost  = round(duration_min * RATE_PER_MIN,  2)
    raw       = round(BASE_FARE + dist_cost + dur_cost, 2)
    m_car     = MULT_CAR.get(car_type, 1.0)

    surge_total = round(ml_surge * m_car, 4)
    final       = round(raw * surge_total, 2)
    min_applied = False
    if final < MIN_FARE:
        final, min_applied = MIN_FARE, True

    # Récupère les labels règles pour l'audit
    rules = compute_price_rules(distance_km, duration_min, row, car_type)
    rules.surge_multiplier = surge_total
    rules.final_price      = final
    rules.min_applied      = min_applied
    rules.ml_used          = True
    rules.source           = f"ML ensemble (XGB×0.55 + LGBM×0.45)  surge_raw={ml_surge:.4f}"
    rules.labels["ml"]     = f"surge_ml={ml_surge:.4f} × car={m_car}"

    return rules


# ══════════════════════════════════════════════════════════════════
# PIPELINE COMPLET  (OSRM + Open-Meteo + Flags + ML → Prix)
# ══════════════════════════════════════════════════════════════════

def calculate_trip_price(
    # Coordonnées GPS
    lat_origin:        float,
    lon_origin:        float,
    lat_dest:          float,
    lon_dest:          float,
    # Contexte ville / zone
    zone_type:         str   = "intérieure",
    has_beach:         int   = 0,
    population:        int   = 300_000,
    intensite_ville:   int   = 3,
    # Conditions de trafic
    trafic_niveau:     int   = 1,
    demande:           str   = "normal",
    indice_congestion: int   = 30,
    retard_estime_min: int   = 5,
    vitesse_moy_kmh:   float = 40.0,
    chauffeurs_actifs: int   = 30,
    # Véhicule
    car_type:          str   = "comfort",
    # Date/heure de réservation (None → maintenant)
    booking_dt:        datetime | None = None,
    # Activer ou non le ML
    use_ml:            bool  = True,
) -> dict:
    """
    Pipeline complet de tarification dynamique Moviroo.

    Étapes :
      1. OSRM        → distance_km, duration_min réels
      2. Open-Meteo  → météo à l'heure exacte de la réservation
      3. Flags        → Ramadan, Jumu'ah, beach surge, période
      4. ML ou règles → surge_multiplier
      5. Prix final   → TND

    Retourne :
        dict complet avec le prix, les détails, les multiplicateurs
        et les données contextuelles (météo, flags).
    """
    if booking_dt is None:
        booking_dt = datetime.now()

    _sep("MOVIROO — Calcul prix dynamique")

    # ── 1. Distance & Durée ───────────────────────────────────────
    _step(1, "Distance OSRM")
    distance_km, duration_min = get_osrm_distance(
        lat_origin, lon_origin, lat_dest, lon_dest
    )
    print(f"     {distance_km:.2f} km  |  {duration_min:.1f} min")

    # ── 2. Météo ──────────────────────────────────────────────────
    _step(2, "Météo Open-Meteo")
    weather = fetch_weather(lat_origin, lon_origin, booking_dt)
    print(
        f"     {weather['weather_label'].capitalize()} | "
        f"{weather['temperature_2m']}°C | "
        f"Vent {weather['windspeed_10m']} km/h | "
        f"is_night={weather['is_night']}"
    )

    # ── 3. Flags temporels ────────────────────────────────────────
    _step(3, "Flags temporels & culturels")
    time_flags  = compute_time_flags(booking_dt)
    beach_flags = compute_beach_flags(has_beach, booking_dt)
    print(f"     Période : {time_flags['periode']}  |  Saison : {time_flags['season']}")
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
        # Météo réelle
        "weather_code":        weather["weather_code"],
        "temperature_2m":      weather["temperature_2m"],
        "windspeed_10m":       weather["windspeed_10m"],
        "precipitation":       weather["precipitation"],
        "is_night":            weather["is_night"],
        # Flags temporels
        **time_flags,
        # Flags beach
        **beach_flags,
    }

    # ── 4. Surge ──────────────────────────────────────────────────
    _step(4, "Calcul du surge")
    ml_result: dict | None = None
    result:    PriceResult

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

    # ── 5. Résultat ───────────────────────────────────────────────
    output = {
        # Prix
        "base_fare":         result.base_fare,
        "distance_cost":     result.distance_cost,
        "duration_cost":     result.duration_cost,
        "raw_price":         result.raw_price,
        "surge_multiplier":  result.surge_multiplier,
        "final_price":       result.final_price,
        "currency":          "TND",
        "min_applied":       result.min_applied,
        # Multiplicateurs
        "mult_traffic":      result.mult_traffic,
        "mult_weather":      result.mult_weather,
        "mult_demand":       result.mult_demand,
        "mult_night":        result.mult_night,
        "mult_car":          result.mult_car,
        "mult_friday":       result.mult_friday,
        "mult_ramadan":      result.mult_ramadan,
        "mult_beach":        result.mult_beach,
        "mult_zone":         result.mult_zone,
        # ML
        "ml_used":           result.ml_used,
        "ml_surge_xgb":      result.ml_surge_xgb,
        "ml_surge_lgbm":     result.ml_surge_lgbm,
        "ml_confidence":     result.ml_confidence,
        # Métadonnées
        "source":            result.source,
        "labels":            result.labels,
        "distance_km":       distance_km,
        "duration_min":      duration_min,
        "car_type":          car_type,
        "zone_type":         zone_type,
        "booking_dt":        booking_dt.isoformat(),
        # Données contextuelles
        "weather":           weather,
        "time_flags":        time_flags,
        "beach_flags":       beach_flags,
    }

    _print_result(output)
    return output


# ══════════════════════════════════════════════════════════════════
# AFFICHAGE RÉCAPITULATIF
# ══════════════════════════════════════════════════════════════════

def _print_result(r: dict) -> None:
    w   = 58
    ml  = r.get("ml_surge_xgb") is not None
    wth = r.get("weather", {})
    tf  = r.get("time_flags", {})

    print("\n┌" + "─"*w + "┐")
    print(f"│  💳  RÉCAPITULATIF COURSE{'':<{w-26}}│")
    print("├" + "─"*w + "┤")
    print(f"│  Distance : {r['distance_km']:.2f} km  /  {r['duration_min']:.1f} min{'':<{w-37}}│")
    print(f"│  Véhicule : {r['car_type']:<{w-13}}│")
    print(f"│  Zone     : {r['zone_type']:<{w-13}}│")
    print(f"│  Heure    : {tf.get('heure_int', '?'):02}h  |  {wth.get('weather_label','?'):<{w-24}}│")
    print("├" + "─"*w + "┤")
    print(f"│  Base     : {BASE_FARE:.2f} TND{'':<{w-21}}│")
    print(f"│  Distance : {r['distance_cost']:.2f} TND{'':<{w-21}}│")
    print(f"│  Durée    : {r['duration_cost']:.2f} TND{'':<{w-21}}│")
    print(f"│  Brut     : {r['raw_price']:.2f} TND{'':<{w-21}}│")
    print("├" + "─"*w + "┤")
    if ml:
        print(f"│  XGB      : ×{r['ml_surge_xgb']:.4f}{'':<{w-18}}│")
        print(f"│  LGBM     : ×{r['ml_surge_lgbm']:.4f}{'':<{w-18}}│")
        conf = r.get('ml_confidence', 0)
        print(f"│  Confiance: {conf*100:.1f}%{'':<{w-15}}│")
    print(f"│  Surge    : ×{r['surge_multiplier']:.4f}{'':<{w-18}}│")
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
