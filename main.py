

from __future__ import annotations

from datetime      import datetime
from models.predictor import predictor
from pricing.engine   import calculate_trip_price



TRIPS = [

    {
        "label":      "hammamet → Aéroport (nuit 00h)",
        "lat_origin":  36.8325, "lon_origin": 10.2330,
        "lat_dest": 36.8785, "lon_dest": 10.3245,
        "zone_type":  "banlieue",
        "has_beach":  0, "population": 497_727, "intensite_ville": 4,
        "trafic_niveau": 1, "demande": "normal",
        "indice_congestion": 50, "retard_estime_min": 1,
        "vitesse_moy_kmh": 37, "chauffeurs_actifs": 44,
        "car_type":   "economy",
        "booking_dt": datetime(2026,4, 14,00, 00),
    },

    

]



if __name__ == "__main__":

    print("=" * 62)
    print("  MOVIROO — Tests de tarification dynamique")
    print(f"  {len(TRIPS)} trajets | OSRM + Open-Meteo + ML")
    print("=" * 62)

    print("\n⏳ Chargement des modèles ML ...")
    predictor.load()

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
# from fastapi import FastAPI
# from datetime import datetime
# from pydantic import BaseModel

# from models.predictor import predictor
# from pricing.engine import calculate_trip_price

# app = FastAPI()

# # Charger modèle au démarrage
# @app.on_event("startup")
# def load_model():
#     print("⏳ Loading ML model...")
#     predictor.load()
#     print("✅ Model loaded")

# # Schéma des données reçues depuis NestJS
# class TripRequest(BaseModel):
#     lat_origin: float
#     lon_origin: float
#     lat_dest: float
#     lon_dest: float
#     zone_type: str
#     has_beach: int
#     population: int
#     intensite_ville: int
#     trafic_niveau: int
#     demande: str
#     indice_congestion: int
#     retard_estime_min: int
#     vitesse_moy_kmh: float
#     chauffeurs_actifs: int
#     car_type: str
#     booking_dt: datetime

# @app.post("/predict-price")
# def predict_price(trip: TripRequest):
#     try:
#         price = calculate_trip_price(**trip.dict())
#         return {"price": price}
#     except Exception as e:
#         return {"error": str(e)}