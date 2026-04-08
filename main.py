

from __future__ import annotations

from datetime      import datetime
from models.predictor import predictor
from pricing.engine   import calculate_trip_price



TRIPS = [

    {
        "label":      "hammamet → Aéroport (nuit 00h)",
        "lat_origin": 36.8625, "lon_origin": 10.1956,
        "lat_dest": 35.8256, "lon_dest": 10.63699,
        "zone_type":  "banlieue",
        "has_beach":  0, "population": 497_727, "intensite_ville": 4,
        "trafic_niveau": 1, "demande": "normal",
        "indice_congestion": 50, "retard_estime_min": 1,
        "vitesse_moy_kmh": 37, "chauffeurs_actifs": 44,
        "car_type":   "economy",
        "booking_dt": datetime(2026,7, 31,13, 0),
    },

    

]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 62)
    print("  MOVIROO — Tests de tarification dynamique")
    print(f"  {len(TRIPS)} trajets | OSRM + Open-Meteo + ML")
    print("=" * 62)

    # Charge le modèle ML (une seule fois)
    print("\n⏳ Chargement des modèles ML ...")
    predictor.load()

    # Lance chaque trajet
    for trip in TRIPS:
        label = trip.pop("label")
        print(f"\n\n🚖  {label}")
        try:
            calculate_trip_price(**trip)
        except Exception as exc:
            print(f"  ❌ Erreur : {exc}")

    print("\n" + "=" * 62)
    print("  Fin des tests")
    print("=" * 62)
