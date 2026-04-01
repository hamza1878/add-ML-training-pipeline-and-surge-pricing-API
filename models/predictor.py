from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy  as np
import pandas as pd
import xgboost  as xgb
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config          import W_XGB, W_LGBM
from models.features import engineer_features, get_feature_list

MODELS_DIR = Path(__file__).parent


class MovirooPredictor:
    """
    Singleton — chargé une seule fois (au démarrage de l'API ou du script).

    Méthodes :
        load()          Charge XGB, LGBM, scaler et feature list depuis models/
        predict(row)    Prédit le surge_multiplier pour un dict contextuel
        is_loaded       bool — True si les modèles sont en mémoire
    """

    def __init__(self):
        self._xgb_model  = None
        self._lgbm_model = None
        self._scaler     = None
        self._features   = None
        self._loaded     = False

    # ─────────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Charge les artefacts ML depuis models/.
        Retourne True si succès, False si les fichiers sont absents.
        """
        xgb_path  = MODELS_DIR / "xgb_model.json"
        lgbm_path = MODELS_DIR / "lgbm_model.txt"
        scal_path = MODELS_DIR / "scaler.pkl"
        feat_path = MODELS_DIR / "feature_columns.json"

        if not xgb_path.exists():
            print(
                f"  ⚠️  Modèle XGB introuvable : {xgb_path}\n"
                "       → Lance d'abord : python models/train.py --csv cleaned_data.csv\n"
                "       → Mode règles métier activé (sans ML)"
            )
            return False

        self._xgb_model = xgb.XGBRegressor()
        self._xgb_model.load_model(str(xgb_path))

        self._lgbm_model = lgb.Booster(model_file=str(lgbm_path))
        self._scaler     = joblib.load(scal_path)

        with open(feat_path) as f:
            self._features = json.load(f)

        self._loaded = True
        print(f"  ✅ ML Predictor chargé — {len(self._features)} features")
        return True

    # ─────────────────────────────────────────────────────────────
    # PROPRIÉTÉ
    # ─────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ─────────────────────────────────────────────────────────────
    # PRÉDICTION
    # ─────────────────────────────────────────────────────────────

    def predict(self, row: dict) -> dict:
        """
        Prédit le surge_multiplier pour un contexte de trajet.

        Paramètres :
            row : dict contenant les colonnes du dataset nettoyé
                  (zone_type, trafic_niveau, demande, heure_int, …)

        Retourne :
            surge_xgb    float   prédiction XGBoost  [1.0 – 3.5]
            surge_lgbm   float   prédiction LightGBM [1.0 – 3.5]
            surge_final  float   ensemble pondéré
            confidence   float   1 - |xgb - lgbm| / mean  ∈ [0, 1]

        Lève :
            RuntimeError  si .load() n'a pas été appelé
        """
        if not self._loaded:
            raise RuntimeError(
                "Predictor non chargé — appelez predictor.load() d'abord"
            )

        # ── Préparation du DataFrame ──────────────────────────────
        df_row = pd.DataFrame([row])

        # Valeurs par défaut pour colonnes manquantes
        _defaults = {
            "heure_int":               row.get("hour_of_day",        12),
            "jour_semaine":            row.get("day_of_week",          0),
            "minute":                  row.get("minute",               0),
            "trafic_niveau":           row.get("trafic_niveau",        1),
            "weather_code":            row.get("weather_code",         1),
            "temperature_2m":          row.get("temperature_2m",    22.0),
            "windspeed_10m":           row.get("windspeed_10m",     10.0),
            "precipitation":           row.get("precipitation",      0.0),
            "is_night":                row.get("is_night",             0),
            # Flags Ramadan
            "is_ramadan_slot":         row.get("is_ramadan_slot",      0),
            "is_ramadan_last_week":    row.get("is_ramadan_last_week", 0),   # NOUVEAU
            # Flags Aïd
            "is_aid_el_fitr":          row.get("is_aid_el_fitr",       0),   # NOUVEAU
            "is_aid_adha_week":        row.get("is_aid_adha_week",     0),   # NOUVEAU
            # Flags Nouvel An
            "is_new_year_eve":         row.get("is_new_year_eve",      0),   # NOUVEAU
            "is_new_year_days":        row.get("is_new_year_days",     0),   # NOUVEAU
            # Autres flags
            "is_friday_slot":          row.get("is_friday_slot",       0),
            "is_school_slot":          row.get("is_school_slot",       0),
            "is_prayer_slot":          row.get("is_prayer_slot",       0),
            "has_beach":               row.get("has_beach",            0),
            "is_beach_hour":           row.get("is_beach_hour",        0),
            "beach_surge_applied":     row.get("beach_surge_applied",  0),
            "beach_surge_value":       row.get("beach_surge_value",   1.0),
            "beach_peak_reason":       row.get("beach_peak_reason", "none"),
            "demande":                 row.get("demande",         "normal"),
            "zone_type":               row.get("zone_type",  "intérieure"),
            # NOUVEAU : 6 types de véhicules (défaut comfort)
            "car_type":                row.get("car_type",       "comfort"),
            "special_event":           row.get("special_event",    "none"),   # NOUVEAU
            "intensite_ville":         row.get("intensite_ville",      3),
            "population":              row.get("population",      300_000),
            "indice_congestion":       row.get("indice_congestion",   30),
            "retard_estime_min":       row.get("retard_estime_min",    5),
            "vitesse_moy_kmh":         row.get("vitesse_moy_kmh",     40),
            "chauffeurs_actifs":       row.get("chauffeurs_actifs",   30),
            "periode":                 row.get("periode", "circulation_normale"),
        }
        for col, val in _defaults.items():
            if col not in df_row.columns:
                df_row[col] = val

        # ── Feature engineering ───────────────────────────────────
        df_eng = engineer_features(df_row)

        # Aligner sur les features exactes du modèle
        features = self._features or get_feature_list()
        for f in features:
            if f not in df_eng.columns:
                df_eng[f] = 0.0

        X   = df_eng[features].astype(float).values
        X_s = self._scaler.transform(X)

        # ── Prédictions individuelles ─────────────────────────────
        s_xgb  = float(self._xgb_model.predict(X_s)[0])
        s_lgbm = float(self._lgbm_model.predict(X_s)[0])

        # Clamp [1.0 – 3.5]
        s_xgb  = max(1.0, min(3.5, s_xgb))
        s_lgbm = max(1.0, min(3.5, s_lgbm))

        # ── Ensemble ──────────────────────────────────────────────
        s_final    = W_XGB * s_xgb + W_LGBM * s_lgbm
        mean_s     = (s_xgb + s_lgbm) / 2
        confidence = max(0.0, 1.0 - abs(s_xgb - s_lgbm) / (mean_s + 1e-9))

        return {
            "surge_xgb":   round(s_xgb,    4),
            "surge_lgbm":  round(s_lgbm,   4),
            "surge_final": round(s_final,  4),
            "confidence":  round(confidence, 4),
        }


predictor = MovirooPredictor()