

from __future__ import annotations

import argparse
import time
import warnings
from datetime import datetime
from pathlib  import Path

import numpy  as np
import pandas as pd

from utils.weather import fetch_weather, WEATHER_LABELS
from utils.flags   import compute_time_flags, compute_beach_flags

warnings.filterwarnings("ignore")

# ── Paramètres par défaut ─────────────────────────────────────────
DEFAULT_INPUT  = "tunisia_all_cities_traffic.csv"
DEFAULT_OUTPUT = "cleaned_data.csv"
DATETIME_COL   = "reservation_datetime"
# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — NETTOYAGE
# ══════════════════════════════════════════════════════════════════

def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
 
    print(f"  Shape initial : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    total_nulls_before = df.isnull().sum().sum()

    # ── Numériques → médiane ──────────────────────────────────────
    for col in df.select_dtypes(include=[np.number]).columns:
        n = df[col].isnull().sum()
        if n:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"    [{col}] {n} null(s) → médiane {med:.3f}")

    # ── Flags booléens → 0 ────────────────────────────────────────
    bool_cols = [
        "has_beach", "is_night", "is_ramadan_slot", "is_friday_slot",
        "is_school_slot", "is_prayer_slot", "is_beach_hour",
        "beach_surge_applied",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── Valeur continue spéciale ──────────────────────────────────
    if "beach_surge_value" in df.columns:
        df["beach_surge_value"] = df["beach_surge_value"].fillna(1.0)

    # ── Colonnes texte → "" ───────────────────────────────────────
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("")

    total_nulls_after = df.isnull().sum().sum()
    print(
        f"  Nulls : {total_nulls_before:,} → {total_nulls_after}  "
        f"| Lignes conservées : {len(df):,} ✅"
    )
    return df


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — PARSING DATETIME
# ══════════════════════════════════════════════════════════════════

def _parse_datetime(df: pd.DataFrame, now_dt: datetime) -> pd.DataFrame:
    """
    Parse la colonne reservation_datetime.
    Si absente, l'injecte depuis now_dt.
    """
    if DATETIME_COL not in df.columns:
        print(f"  ℹ️  '{DATETIME_COL}' absente → injectée depuis now_dt")
        df[DATETIME_COL] = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df[DATETIME_COL] = df[DATETIME_COL].fillna(pd.Timestamp(now_dt))

    print(f"  Plage : {df[DATETIME_COL].min()} → {df[DATETIME_COL].max()}")
    return df


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — FLAGS TEMPORELS
# ══════════════════════════════════════════════════════════════════

def _add_time_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les flags temporels (Ramadan, Vendredi, période, …)
    depuis la date réelle de chaque ligne.
    """
    time_df = pd.DataFrame(
        df[DATETIME_COL]
        .apply(lambda ts: compute_time_flags(ts.to_pydatetime()))
        .tolist(),
        index=df.index,
    )
    for col in time_df.columns:
        df[col] = time_df[col]

    print(f"  Ramadan  : {df['is_ramadan_slot'].sum():,} lignes")
    print(f"  Vendredi : {df['is_friday_slot'].sum():,} lignes")
    print(f"  Saisons  : {df['season'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 4 — FLAGS BEACH
# ══════════════════════════════════════════════════════════════════

def _add_beach_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les flags beach (saison-aware) depuis la date réelle.
    """
    if "has_beach" not in df.columns:
        print("  ⚠️  'has_beach' absente — beach flags ignorés")
        return df

    beach_df = pd.DataFrame(
        df.apply(
            lambda row: compute_beach_flags(
                int(row["has_beach"]),
                row[DATETIME_COL].to_pydatetime(),
            ),
            axis=1,
        ).tolist(),
        index=df.index,
    )
    for col in beach_df.columns:
        df[col] = beach_df[col]

    print(f"  Surge actif : {df['beach_surge_applied'].sum():,} lignes")
    print(df["beach_peak_reason"].value_counts().to_string())
    return df


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 5 — MÉTÉO OPEN-METEO
# ══════════════════════════════════════════════════════════════════

def _add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit chaque ligne avec la météo réelle Open-Meteo.
    Cache (lat, lon, date) pour minimiser les appels API.
    """
    print(f"  → {len(df):,} lignes à enrichir")
    _cache: dict = {}
    weather_rows = []

    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"     {i+1:,}/{len(df):,}")

        lat    = round(float(row["latitude"]),  4)
        lon    = round(float(row["longitude"]), 4)
        row_dt = row[DATETIME_COL].to_pydatetime()
        key    = (lat, lon, row_dt.strftime("%Y-%m-%d"))

        if key not in _cache:
            _cache[key] = fetch_weather(lat, lon, row_dt)
            time.sleep(0.05)   # respecter le rate-limit Open-Meteo

        weather_rows.append(_cache[key])

    weather_df = pd.DataFrame(weather_rows, index=df.index)
    for col in weather_df.columns:
        df[col] = weather_df[col]

    # ── Validation weather_code ───────────────────────────────────
    bad = ~df["weather_code"].isin([1, 2, 3, 4])
    if bad.sum():
        print(f"  ⚠️  {bad.sum()} weather_code invalides → forcé à 1 (clair)")
        df.loc[bad, "weather_code"] = 1

    print("\n  Distribution weather_code :")
    for code in sorted(df["weather_code"].unique()):
        label = WEATHER_LABELS.get(int(code), "?")
        count = (df["weather_code"] == code).sum()
        print(f"     {int(code)} ({label:<8}) : {count:>6} lignes")

    print("\n  is_night :")
    for val, cnt in df["is_night"].value_counts().sort_index().items():
        label = "Nuit 🌙" if val else "Jour ☀️"
        print(f"     {label} : {cnt:,}")

    return df


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 6 — ORDRE DES COLONNES
# ══════════════════════════════════════════════════════════════════

_COL_ORDER = [
    DATETIME_COL, "reservation_date", "reservation_time",
    "heure", "heure_int", "minute", "jour_semaine", "jour_nom", "season",
    "ville", "gouvernorat", "zone_type",
    "latitude", "longitude", "population", "intensite_ville",
    "has_beach", "beach_name", "periode", "cause_circulation",
    # Météo
    "weather_code", "weathercode_raw", "weather_label", "weather_mult",
    "temperature_2m", "precipitation", "rain",
    "windspeed_10m", "visibility", "sunrise", "sunset",
    # Trafic
    "trafic_niveau", "trafic_label", "demande",
    "indice_congestion", "retard_estime_min", "vitesse_moy_kmh", "chauffeurs_actifs",
    # Flags temporels / culturels
    "is_night", "is_ramadan_slot", "is_friday_slot",
    "is_school_slot", "is_prayer_slot",
    # Flags beach
    "is_beach_hour", "beach_peak_reason", "beach_surge_applied", "beach_surge_value",
    # Cible ML
    "surge_multiplier",
]


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in _COL_ORDER if c in df.columns]
    extra   = [c for c in df.columns if c not in _COL_ORDER]
    return df[present + extra]


# ══════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def run_pipeline(
    input_csv:  str = DEFAULT_INPUT,
    output_csv: str = DEFAULT_OUTPUT,
    now_dt:     datetime | None = None,
) -> pd.DataFrame:
    """
    Exécute le pipeline complet de nettoyage et d'enrichissement.

    Paramètres :
        input_csv   CSV brut en entrée (colonnes du dataset Moviroo)
        output_csv  CSV nettoyé en sortie  → prêt pour train.py
        now_dt      datetime de référence (None → datetime.now())

    Retourne :
        pd.DataFrame du CSV nettoyé
    """
    if now_dt is None:
        now_dt = datetime.now()

    _sep("MOVIROO — Pipeline Nettoyage + Enrichissement Météo")
    print(f"  Référence temps : {now_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Chargement ────────────────────────────────────────────────
    df = pd.read_csv(input_csv)
    print(f"\n✅ CSV chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # ── Étapes ────────────────────────────────────────────────────
    _section("Étape 1 — Nettoyage")
    df = _clean_dataset(df)

    _section("Étape 2 — Parsing datetime")
    df = _parse_datetime(df, now_dt)

    _section("Étape 3 — Flags temporels")
    df = _add_time_flags(df)

    _section("Étape 4 — Flags beach")
    df = _add_beach_flags(df)

    _section("Étape 5 — Météo Open-Meteo")
    df = _add_weather(df)

    _section("Étape 6 — Ordre des colonnes")
    df = _reorder_columns(df)
    print(f"  {df.shape[1]} colonnes ordonnées")

    # ── Sauvegarde ────────────────────────────────────────────────
    df.to_csv(output_csv, index=False)
    _sep()
    print(f"✅ Sauvegardé → {output_csv}")
    print(f"   {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"   Nulls restants : {df.isnull().sum().sum()}")

    return df


# ══════════════════════════════════════════════════════════════════
# HELPERS D'AFFICHAGE
# ══════════════════════════════════════════════════════════════════

def _sep(title: str = "") -> None:
    print("\n" + "=" * 62)
    if title:
        print(f"  {title}")
        print("=" * 62)


def _section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Moviroo — Pipeline de nettoyage et enrichissement CSV"
    )
    parser.add_argument(
        "--input",  default=DEFAULT_INPUT,
        help=f"CSV brut en entrée (défaut : {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"CSV nettoyé en sortie (défaut : {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    df_final = run_pipeline(input_csv=args.input, output_csv=args.output)

    # ── Aperçu final ──────────────────────────────────────────────
    preview_cols = [
        DATETIME_COL, "season", "ville", "is_night",
        "weather_code", "weather_label", "temperature_2m",
        "is_ramadan_slot", "is_friday_slot",
        "beach_surge_applied", "beach_peak_reason",
        "surge_multiplier",
    ]
    preview = [c for c in preview_cols if c in df_final.columns]
    print("\n--- APERÇU FINAL (5 premières lignes) ---")
    print(df_final[preview].head(5).to_string(index=False))
