

from __future__ import annotations

import numpy  as np
import pandas as pd

from config import (
    ZONE_MAP, DEMAND_MAP, PERIODE_MAP, BEACH_REASON_MAP, CAR_MAP,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique toutes les transformations nécessaires pour préparer
    les features du modèle ML à partir du CSV nettoyé.

    Transformations :
      • Encodage ordinal des catégorielles (zone_type, demande, …)
      • Heure & jour sous forme cyclique (sin/cos)
      • Interactions entre variables contextuelles
      • Population en log1p
      • Inverses (vitesse, chauffeurs) pour les relations non-linéaires

    Idempotent — peut être appelé plusieurs fois sans effets de bord.
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

    # car_type_code — absent dans certains CSV → défaut comfort=2
    if "car_type_code" not in df.columns:
        if "car_type" in df.columns:
            df["car_type_code"] = (
                df["car_type"].astype(str).str.lower()
                .map(CAR_MAP).fillna(2).astype(int)
            )
        else:
            df["car_type_code"] = 2

    # ── jour_semaine — absent dans certains CSV ───────────────────
    if "jour_semaine" not in df.columns:
        df["jour_semaine"] = 0

    # ── Cycliques heure ───────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["heure_int"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["heure_int"] / 24)

    # ── Cycliques jour ────────────────────────────────────────────
    df["day_sin"] = np.sin(2 * np.pi * df["jour_semaine"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["jour_semaine"] / 7)

    # ── Cycliques minute ──────────────────────────────────────────
    if "minute" not in df.columns:
        df["minute"] = 0
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    # ── Interactions ──────────────────────────────────────────────
    df["traffic_x_demand"]    = df["trafic_niveau"]       * df["demande_enc"]
    df["beach_x_hour"]        = df["has_beach"]            * df["is_beach_hour"]
    df["night_x_traffic"]     = df["is_night"]             * df["trafic_niveau"]
    df["ramadan_x_traffic"]   = df["is_ramadan_slot"]      * df["trafic_niveau"]
    df["beach_x_surge"]       = df["beach_surge_applied"]  * df["beach_surge_value"]
    df["congestion_x_retard"] = df["indice_congestion"]    * df["retard_estime_min"]

    # ── Inverses ──────────────────────────────────────────────────
    df["vitesse_inv"]    = 1.0 / (df["vitesse_moy_kmh"]   + 1)
    df["chauffeurs_inv"] = 1.0 / (df["chauffeurs_actifs"]  + 1)

    # ── Population log ────────────────────────────────────────────
    df["population_log"] = np.log1p(df["population"])

    return df


def get_feature_list() -> list[str]:
    """
    Retourne la liste ordonnée des colonnes attendues par le modèle.
    Doit être identique entre train.py et predictor.py.
    """
    return [
        # ── Trafic ─────────────────────────────────────────────
        "trafic_niveau",
        "indice_congestion",
        "retard_estime_min",
        "vitesse_moy_kmh",
        "chauffeurs_actifs",
        # ── Temporel cyclique ───────────────────────────────────
        "heure_int",
        "minute",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "minute_sin",
        "minute_cos",
        # ── Flags booléens ──────────────────────────────────────
        "is_night",
        "is_ramadan_slot",
        "is_friday_slot",
        "is_school_slot",
        "is_prayer_slot",
        "is_beach_hour",
        "beach_surge_applied",
        # ── Valeurs continues ───────────────────────────────────
        "beach_surge_value",
        "has_beach",
        # ── Encodages catégoriels ───────────────────────────────
        "zone_type_enc",
        "demande_enc",
        "periode_enc",
        "beach_reason_enc",
        "car_type_code",
        # ── Géo / Ville ─────────────────────────────────────────
        "intensite_ville",
        "population_log",
        # ── Interactions ────────────────────────────────────────
        "traffic_x_demand",
        "beach_x_hour",
        "night_x_traffic",
        "ramadan_x_traffic",
        "beach_x_surge",
        "congestion_x_retard",
        "vitesse_inv",
        "chauffeurs_inv",
    ]
