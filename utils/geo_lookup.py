"""
utils/geo_lookup.py
───────────────────
Recherche le point le plus proche dans le dataset CSV
pour une paire (lat, lon) donnée, dans un rayon de 20 km.

Retourne les métadonnées du point trouvé (ville, zone_type,
has_beach, population, intensite_ville, …) ou None si rien
dans le rayon.

Usage :
    from utils.geo_lookup import DatasetLookup
    lookup = DatasetLookup("cleaned_data.csv")  # chargé une fois
    meta   = lookup.find_nearest(36.8625, 10.1956, radius_km=20)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing  import Optional

import pandas as pd


# ── Colonnes à retourner depuis le dataset ────────────────────────
_META_COLS = [
    "ville", "gouvernorat", "zone_type",
    "latitude", "longitude",
    "population", "intensite_ville",
    "has_beach", "beach_name",
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance haversine en km entre deux points GPS."""
    R = 6_371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


class DatasetLookup:
    """
    Charge le dataset une seule fois et expose find_nearest().

    Paramètres :
        csv_path   chemin vers cleaned_data.csv (ou le CSV brut)
        radius_km  rayon de recherche par défaut (20 km)
    """

    def __init__(self, csv_path: str, radius_km: float = 20.0):
        path = Path(csv_path)
        if not path.exists():
            print(f"  ⚠️  DatasetLookup : fichier '{csv_path}' introuvable — lookup désactivé")
            self._df      = None
            self._points  = []
            return

        df = pd.read_csv(csv_path, usecols=lambda c: c in _META_COLS + ["latitude", "longitude"])

        # Dédupliquer par (latitude, longitude) — on veut les métadonnées uniques
        keep = [c for c in _META_COLS if c in df.columns]
        df   = df[keep].drop_duplicates(subset=["latitude", "longitude"])

        self._df     = df.reset_index(drop=True)
        self._points = list(zip(df["latitude"], df["longitude"]))
        self._radius = radius_km
        print(f"  📍 DatasetLookup chargé : {len(self._points)} points uniques (rayon {radius_km} km)")

    # ─────────────────────────────────────────────────────────────

    def find_nearest(
        self,
        lat: float,
        lon: float,
        radius_km: float | None = None,
    ) -> Optional[dict]:
        """
        Cherche le point du dataset le plus proche de (lat, lon).

        Retourne un dict avec les métadonnées du point si la distance
        est ≤ radius_km, sinon None.

        Le dict contient :
            ville, gouvernorat, zone_type,
            population, intensite_ville,
            has_beach, beach_name,
            latitude_dataset, longitude_dataset,
            distance_km          (distance réelle au point trouvé)
            in_dataset           True
        """
        if self._df is None or not self._points:
            return None

        r = radius_km if radius_km is not None else self._radius

        best_dist = float("inf")
        best_idx  = -1

        for i, (plat, plon) in enumerate(self._points):
            d = _haversine_km(lat, lon, plat, plon)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        if best_dist > r:
            print(
                f"  📍 Lookup ({lat:.4f}, {lon:.4f}) : "
                f"point le plus proche à {best_dist:.1f} km > {r} km → hors dataset"
            )
            return None

        row = self._df.iloc[best_idx].to_dict()
        row["latitude_dataset"] = row.pop("latitude", lat)
        row["longitude_dataset"] = row.pop("longitude", lon)
        row["distance_km"]      = round(best_dist, 2)
        row["in_dataset"]       = True

        # Valeurs par défaut pour colonnes absentes
        row.setdefault("ville",           "Inconnue")
        row.setdefault("gouvernorat",     "Inconnu")
        row.setdefault("zone_type",       "intérieure")
        row.setdefault("population",      300_000)
        row.setdefault("intensite_ville", 3)
        row.setdefault("has_beach",       0)
        row.setdefault("beach_name",      "")

        print(
            f"  📍 Lookup ({lat:.4f}, {lon:.4f}) → "
            f"{row['ville']} / {row['zone_type']} "
            f"(dist={best_dist:.1f} km, beach={int(row['has_beach'])})"
        )
        return row

    # ─────────────────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._df is not None
