from __future__ import annotations

import numpy  as np
import pandas as pd

from config import (
    ZONE_MAP, DEMAND_MAP, PERIODE_MAP, BEACH_REASON_MAP, CAR_MAP,
)

SPECIAL_EVENT_MAP: dict[str, int] = {
    "none":             0,
    "new_year_days":    1,
    "new_year_eve":     2,
    "aid_adha_week":    3,
    "aid_el_adha_week": 3,
    "aid_el_fitr":      4,
    "ramadan_last_week":5,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les features ML.

    CORRECTIF CLÉ : weather_code, temperature_2m, windspeed_10m et
    precipitation sont désormais des features directes du modèle.
    Ainsi pluie (code=2), tempête (code=3) et sirocco (code=4)
    produisent des prédictions différentes de clair (code=1).
    Sans ces colonnes le ML ne voyait jamais la météo.
    """
    df = df.copy()

    # ── Encodages catégoriels ─────────────────────────────────────
    df["zone_type_enc"] = (
        df["zone_type"].astype(str).str.lower().str.strip()
        .map(ZONE_MAP).fillna(3).astype(int)
    )
    df["demande_enc"] = (
        df["demande"].astype(str).str.lower().str.strip()
        .map(DEMAND_MAP).fillna(0).astype(int)
    )
    df["periode_enc"] = (
        df["periode"].astype(str).str.lower().str.strip()
        .map(PERIODE_MAP).fillna(1).astype(int)
    )
    df["beach_reason_enc"] = (
        df["beach_peak_reason"].fillna("").astype(str).str.lower().str.strip()
        .map(BEACH_REASON_MAP).fillna(0).astype(int)
    )

    # ── Événements spéciaux ───────────────────────────────────────
    if "special_event" in df.columns:
        df["special_event_enc"] = (
            df["special_event"].fillna("none").astype(str).str.lower().str.strip()
            .map(SPECIAL_EVENT_MAP).fillna(0).astype(int)
        )
    else:
        df["special_event_enc"] = 0

    # ── Véhicule — 6 types ────────────────────────────────────────
    if "car_type_code" not in df.columns:
        if "car_type" in df.columns:
            df["car_type_code"] = (
                df["car_type"].astype(str).str.lower()
                .str.replace(" ", "_")
                .map(CAR_MAP).fillna(3).astype(int)
            )
        else:
            df["car_type_code"] = 3

    # ── Colonnes manquantes avec valeurs neutres ──────────────────
    if "jour_semaine" not in df.columns:
        df["jour_semaine"] = 0
    if "minute" not in df.columns:
        df["minute"] = 0

    # Météo — valeurs neutres si absentes
    if "weather_code"   not in df.columns: df["weather_code"]   = 1
    if "temperature_2m" not in df.columns: df["temperature_2m"] = 22.0
    if "windspeed_10m"  not in df.columns: df["windspeed_10m"]  = 10.0
    if "precipitation"  not in df.columns: df["precipitation"]  = 0.0

    # ── Cycliques ─────────────────────────────────────────────────
    df["hour_sin"]   = np.sin(2 * np.pi * df["heure_int"]    / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["heure_int"]    / 24)
    df["day_sin"]    = np.sin(2 * np.pi * df["jour_semaine"] / 7)
    df["day_cos"]    = np.cos(2 * np.pi * df["jour_semaine"] / 7)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"]       / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"]       / 60)

    # ── Nouveaux flags manquants → 0 ──────────────────────────────
    for col in [
        "is_ramadan_last_week", "is_aid_el_fitr",
        "is_aid_adha_week", "is_new_year_eve", "is_new_year_days",
    ]:
        if col not in df.columns:
            df[col] = 0

    # ── Interactions ──────────────────────────────────────────────
    df["traffic_x_demand"]       = df["trafic_niveau"]       * df["demande_enc"]
    df["beach_x_hour"]           = df["has_beach"]            * df["is_beach_hour"]
    df["night_x_traffic"]        = df["is_night"]             * df["trafic_niveau"]
    df["ramadan_x_traffic"]      = df["is_ramadan_slot"]      * df["trafic_niveau"]
    df["beach_x_surge"]          = df["beach_surge_applied"]  * df["beach_surge_value"]
    df["congestion_x_retard"]    = df["indice_congestion"]    * df["retard_estime_min"]
    df["special_x_traffic"]      = df["special_event_enc"]   * df["trafic_niveau"]
    df["aid_fitr_x_hour"]        = df["is_aid_el_fitr"]      * df["heure_int"]
    df["ram_lastweek_x_traffic"] = df["is_ramadan_last_week"] * df["trafic_niveau"]
    # Météo × contexte : pluie + nuit / pluie + trafic
    df["weather_x_traffic"]      = df["weather_code"]         * df["trafic_niveau"]
    df["weather_x_night"]        = df["weather_code"]         * df["is_night"]

    # ── Inverses ──────────────────────────────────────────────────
    df["vitesse_inv"]    = 1.0 / (df["vitesse_moy_kmh"]  + 1)
    df["chauffeurs_inv"] = 1.0 / (df["chauffeurs_actifs"] + 1)

    # ── Population log ────────────────────────────────────────────
    df["population_log"] = np.log1p(df["population"])

    return df


def get_feature_list() -> list[str]:
    """
    Liste ordonnée identique entre train.py et predictor.py.

    Les 4 features météo directes permettent au modèle de
    distinguer clair / pluie / tempête / sirocco lors de l'inférence.
    """
    return [
        # Trafic
        "trafic_niveau",
        "indice_congestion",
        "retard_estime_min",
        "vitesse_moy_kmh",
        "chauffeurs_actifs",
        # Météo directe (AJOUT — différencie les conditions météo)
        "weather_code",
        "temperature_2m",
        "windspeed_10m",
        "precipitation",
        # Temporel cyclique
        "heure_int",
        "minute",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "minute_sin",
        "minute_cos",
        # Flags booléens
        "is_night",
        "is_ramadan_slot",
        "is_ramadan_last_week",
        "is_aid_el_fitr",
        "is_aid_adha_week",
        "is_new_year_eve",
        "is_new_year_days",
        "is_friday_slot",
        "is_school_slot",
        "is_prayer_slot",
        "is_beach_hour",
        "beach_surge_applied",
        # Continues
        "beach_surge_value",
        "has_beach",
        # Encodages
        "zone_type_enc",
        "demande_enc",
        "periode_enc",
        "beach_reason_enc",
        "car_type_code",
        "special_event_enc",
        # Géo
        "intensite_ville",
        "population_log",
        # Interactions
        "traffic_x_demand",
        "beach_x_hour",
        "night_x_traffic",
        "ramadan_x_traffic",
        "beach_x_surge",
        "congestion_x_retard",
        "special_x_traffic",
        "aid_fitr_x_hour",
        "ram_lastweek_x_traffic",
        "weather_x_traffic",
        "weather_x_night",
        "vitesse_inv",
        "chauffeurs_inv",
    ]