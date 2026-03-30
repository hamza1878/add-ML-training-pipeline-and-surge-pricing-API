

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy  as np
import pandas as pd
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
import xgboost  as xgb
import lightgbm as lgb

# ── Chemins ───────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # pour importer config

from config        import TARGET_COL, RANDOM_STATE, XGB_PARAMS, LGBM_PARAMS
from models.features import engineer_features, get_feature_list

warnings.filterwarnings("ignore")
import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# csv_full_path = os.path.join(BASE_DIR, "data", "tunisia_all_cities_traffic.csv")
MODELS_DIR = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════════

def _print_metrics(y_true, y_pred, label: str) -> dict:
    """Calcule et affiche MAE, RMSE, R², MAPE. Retourne un dict."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100

    print(
        f"  [{label:<22}]  "
        f"MAE={mae:.4f}  RMSE={rmse:.4f}  "
        f"R²={r2:.4f}  MAPE={mape:.2f}%"
    )
    return {
        "mae":  float(mae),
        "rmse": float(rmse),
        "r2":   float(r2),
        "mape": float(mape),
    }


# ══════════════════════════════════════════════════════════════════
# PIPELINE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════

def train(
    csv_path:   str,
    test_size:  float = 0.15,
    val_size:   float = 0.15,
) -> dict:
    
    """
    Pipeline complet d'entraînement :
      1. Charge le CSV nettoyé
      2. Feature engineering (models/features.py)
      3. Split 70 / val / test
      4. StandardScaler
      5. XGBoost + LightGBM
      6. Ensemble XGB×0.55 + LGBM×0.45
      7. Sauvegarde artefacts dans models/

    Paramètres :
        csv_path    chemin vers cleaned_data.csv
        test_size   fraction du dataset réservée au test  (défaut 0.15)
        val_size    fraction réservée à la validation     (défaut 0.15)

    Retourne :
        dict rapport complet des métriques
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    _sep("MOVIROO — ML Training Pipeline")

    # ── Chargement ────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"\n✅ Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Colonne cible '{TARGET_COL}' absente.\n"
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # ── Feature Engineering ───────────────────────────────────────
    _section("Feature Engineering")
    df_eng = engineer_features(df)

    FEATURES = [f for f in get_feature_list() if f in df_eng.columns]
    missing  = [f for f in get_feature_list() if f not in df_eng.columns]
    print(f"  Features utilisées : {len(FEATURES)}")
    if missing:
        print(f"  ⚠️  Absentes (ignorées) : {missing}")

    X = df_eng[FEATURES].astype(float).fillna(0)
    y = df_eng[TARGET_COL].astype(float)
    print(
        f"  Cible : min={y.min():.3f}  max={y.max():.3f}  "
        f"mean={y.mean():.3f}  std={y.std():.3f}"
    )

    # ── Split ─────────────────────────────────────────────────────
    val_adj = val_size / (1 - test_size)
    X_tmp, X_test,  y_tmp, y_test  = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_adj, random_state=RANDOM_STATE
    )
    print(f"\n  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # ── Scaling ───────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── XGBoost ───────────────────────────────────────────────────
    _section("XGBoost")
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=200,
    )
    xgb_val  = _print_metrics(y_val,  xgb_model.predict(X_val_s),  "XGB Val")
    xgb_test = _print_metrics(y_test, xgb_model.predict(X_test_s), "XGB Test")

    # ── LightGBM ──────────────────────────────────────────────────
    _section("LightGBM")
    lgbm_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    lgbm_model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)])
    lgbm_val  = _print_metrics(y_val,  lgbm_model.predict(X_val_s),  "LGBM Val")
    lgbm_test = _print_metrics(y_test, lgbm_model.predict(X_test_s), "LGBM Test")

    # ── Ensemble 55 % / 45 % ─────────────────────────────────────
    _section("Ensemble (XGB×55% + LGBM×45%)")
    ens_pred = (
        0.55 * xgb_model.predict(X_test_s)
        + 0.45 * lgbm_model.predict(X_test_s)
    )
    ens_test = _print_metrics(y_test, ens_pred, "Ensemble Test")

    # ── Feature Importance (Top 15) ───────────────────────────────
    _section("Top 15 features (XGBoost)")
    fi = pd.Series(xgb_model.feature_importances_, index=FEATURES)
    for feat, imp in fi.sort_values(ascending=False).head(15).items():
        bar = "█" * int(imp * 400)
        print(f"  {feat:<30} {bar} ({imp:.4f})")

    # ── Sauvegarde ────────────────────────────────────────────────
    _section("Sauvegarde des artefacts")
    xgb_model.save_model(str(MODELS_DIR / "xgb_model.json"))
    lgbm_model.booster_.save_model(str(MODELS_DIR / "lgbm_model.txt"))
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURES, f, indent=2)

    print(f"\n✅ Artefacts dans : {MODELS_DIR.resolve()}")
    print("   xgb_model.json | lgbm_model.txt | scaler.pkl | feature_columns.json")

    # ── Rapport ───────────────────────────────────────────────────
    report = {
        "csv_path":      csv_path,
        "n_samples":     int(len(df)),
        "features":      FEATURES,
        "n_features":    len(FEATURES),
        "target":        TARGET_COL,
        "n_train":       int(len(X_train)),
        "n_val":         int(len(X_val)),
        "n_test":        int(len(X_test)),
        "xgb_val":       xgb_val,
        "xgb_test":      xgb_test,
        "lgbm_val":      lgbm_val,
        "lgbm_test":     lgbm_test,
        "ensemble_test": ens_test,
    }
    with open(MODELS_DIR / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    _sep("Entraînement terminé")
    return report


# ══════════════════════════════════════════════════════════════════
# HELPERS D'AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def _sep(title: str = "") -> None:
    print("\n" + "=" * 62)
    if title:
        print(f"  {title}")
        print("=" * 62)


def _section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 58 - len(title)))


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moviroo ML Training")
    parser.add_argument(
        "--csv", default="cleaned_data.csv",
        help="Chemin vers le CSV nettoyé (défaut: cleaned_data.csv)"
    )
    parser.add_argument(
        "--test", type=float, default=0.15,
        help="Fraction réservée au test (défaut: 0.15)"
    )
    parser.add_argument(
        "--val", type=float, default=0.15,
        help="Fraction réservée à la validation (défaut: 0.15)"
    )
    args = parser.parse_args()

    train(csv_path=args.csv, test_size=args.test, val_size=args.val)
