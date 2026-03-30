"""
╔══════════════════════════════════════════════════════════════════╗
║   MOVIROO — utils/weather.py                                     ║
║   Responsabilité : météo réelle via Open-Meteo                   ║
║                                                                  ║
║   Fonctions publiques :                                          ║
║     fetch_weather(lat, lon, target_dt) → dict météo enrichi      ║
║     wmo_to_pricing_code(wmo_code)      → code Moviroo 1-4        ║
║     detect_sirocco(temp, wind, vis, rain) → bool                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from datetime import datetime

import numpy  as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

from config import WEATHER_LABELS, MULT_WEATHER

# ══════════════════════════════════════════════════════════════════
# CLIENT OPEN-METEO  (cache disque 1 h + retry automatique)
# ══════════════════════════════════════════════════════════════════

_cache_session   = requests_cache.CachedSession(".cache_meteo", expire_after=3_600)
_retry_session   = retry(_cache_session, retries=5, backoff_factor=0.2)
_openmeteo_client = openmeteo_requests.Client(session=_retry_session)

# Valeurs par défaut si l'API est inaccessible
_DEFAULT_WEATHER: dict = {
    "temperature_2m":  22.0,
    "precipitation":    0.0,
    "rain":             0.0,
    "windspeed_10m":   10.0,
    "weathercode_raw":  0,
    "visibility":   10_000.0,
    "weather_code":     1,
    "weather_label": "clair",
    "weather_mult":    1.00,
    "sunrise":       "06:00",
    "sunset":        "19:30",
    "is_night":         0,
}


# ══════════════════════════════════════════════════════════════════
# CONVERSION WMO → CODE MOVIROO (1 / 2 / 3 / 4)
# ══════════════════════════════════════════════════════════════════

def wmo_to_pricing_code(wmo_code: float) -> int:
    """
    Convertit un code WMO (retourné par Open-Meteo) en code Moviroo :
      1 = clair    (×1.00)  — dégagé, nuageux sans précip, brouillard
      2 = pluie    (×1.10)  — bruine, pluie, neige légère, averses mod.
      3 = tempête  (×1.30)  — averses fortes, orages
      4 = sirocco  (×1.10)  — détecté par detect_sirocco()
    """
    if pd.isna(wmo_code):
        return 1

    c = int(wmo_code)

    if c in (0, 1, 2, 3):         return 1   # clair / couvert
    if c in (45, 48):              return 1   # brouillard
    if 51 <= c <= 57:              return 2   # bruine
    if 61 <= c <= 67:              return 2   # pluie
    if 71 <= c <= 77:              return 2   # neige
    if 80 <= c <= 82:              return 2   # averses légères/modérées
    if c in (83, 84, 85, 86):     return 3   # averses fortes
    if c == 95:                    return 3   # orage
    if c in (96, 99):             return 3   # orage + grêle
    return 1                                  # inconnu → clair


# ══════════════════════════════════════════════════════════════════
# DÉTECTION SIROCCO
# ══════════════════════════════════════════════════════════════════

def detect_sirocco(
    temp:       float,
    wind:       float,
    visibility: float,
    rain:       float,
) -> bool:
    """
    Détecte le sirocco tunisien (vent chaud et sec du Sahara).
    Conditions requises (toutes simultanément) :
      • température  > 35 °C
      • vent         > 40 km/h
      • visibilité   < 2 000 m  (sable en suspension)
      • pluie        = 0 mm
    """
    return (
        temp > 35
        and wind > 40
        and visibility < 2_000
        and rain == 0.0
    )


# ══════════════════════════════════════════════════════════════════
# APPEL OPEN-METEO
# ══════════════════════════════════════════════════════════════════

def fetch_weather(
    lat:       float,
    lon:       float,
    target_dt: datetime,
) -> dict:
    """
    Récupère la météo réelle pour (lat, lon) à l'heure de target_dt.

    Choisit automatiquement :
      • API archive  si target_dt > 3 jours dans le passé
      • API forecast sinon

    Retourne un dict avec :
        weather_code   int   1-4  (code Moviroo, PAS le WMO brut)
        weathercode_raw int       WMO brut pour audit/debug
        weather_label  str
        weather_mult   float
        temperature_2m float  °C
        precipitation  float  mm
        rain           float  mm
        windspeed_10m  float  km/h
        visibility     float  m
        sunrise        str    "HH:MM"
        sunset         str    "HH:MM"
        is_night       int    0/1  (basé sur sunrise/sunset réels)
    """
    h        = target_dt.hour
    date_str = target_dt.strftime("%Y-%m-%d")
    default  = {**_DEFAULT_WEATHER, "is_night": int(h < 6 or h >= 20)}

    try:
        params = {
            "latitude":  lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m", "precipitation", "rain",
                "windspeed_10m",  "weathercode",   "visibility",
            ],
            "daily":    ["sunrise", "sunset"],
            "timezone": "Africa/Tunis",
        }

        # Choix archive vs forecast
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=3)
        if pd.Timestamp(target_dt) < cutoff:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params["start_date"] = date_str
            params["end_date"]   = date_str
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            params["forecast_days"] = 1

        responses = _openmeteo_client.weather_api(url, params=params)
        resp      = responses[0]
        hourly    = resp.Hourly()

        # ── Extraction de l'heure cible ───────────────────────────
        temp       = float(hourly.Variables(0).ValuesAsNumpy()[h])
        precip     = float(hourly.Variables(1).ValuesAsNumpy()[h])
        rain       = float(hourly.Variables(2).ValuesAsNumpy()[h])
        wind       = float(hourly.Variables(3).ValuesAsNumpy()[h])
        wmo        = float(hourly.Variables(4).ValuesAsNumpy()[h])
        visibility = float(hourly.Variables(5).ValuesAsNumpy()[h])

        # ── Sunrise / Sunset → is_night précis ────────────────────
        daily   = resp.Daily()
        sunrise = (
            pd.to_datetime(daily.Variables(0).ValuesInt64AsNumpy()[0],
                           unit="s", utc=True)
            .tz_convert("Africa/Tunis")
        )
        sunset = (
            pd.to_datetime(daily.Variables(1).ValuesInt64AsNumpy()[0],
                           unit="s", utc=True)
            .tz_convert("Africa/Tunis")
        )
        is_night = int(h < sunrise.hour or h >= sunset.hour)

        # ── Conversion WMO → code Moviroo ─────────────────────────
        wcode = wmo_to_pricing_code(wmo)
        if detect_sirocco(temp, wind, visibility, rain):
            wcode = 4

        return {
            "temperature_2m":  round(temp,       1),
            "precipitation":   round(precip,      2),
            "rain":            round(rain,         2),
            "windspeed_10m":   round(wind,         1),
            "weathercode_raw": int(wmo),
            "visibility":      round(visibility,   0),
            "weather_code":    wcode,
            "weather_label":   WEATHER_LABELS[wcode],
            "weather_mult":    MULT_WEATHER[wcode],
            "sunrise":         sunrise.strftime("%H:%M"),
            "sunset":          sunset.strftime("%H:%M"),
            "is_night":        is_night,
        }

    except Exception as exc:
        print(
            f"  ⚠️  Open-Meteo ({lat:.4f}, {lon:.4f}) @ {date_str} "
            f"→ valeurs par défaut. Erreur : {exc}"
        )
        return default
