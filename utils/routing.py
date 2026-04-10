

# from __future__ import annotations

# import requests

# # ── URL de l'API OSRM publique (remplaçable par une instance privée) ──
# OSRM_BASE_URL = "http://router.project-osrm.org"


# def get_osrm_distance(
#     lat1: float,
#     lon1: float,
#     lat2: float,
#     lon2: float,
#     osrm_url: str = OSRM_BASE_URL,
#     timeout:  int = 10,
# ) -> tuple[float, float]:
   
#     endpoint = (
#         f"{osrm_url}/route/v1/driving/"
#         f"{lon1},{lat1};{lon2},{lat2}"
#         f"?overview=false"
#     )

#     try:
#         resp = requests.get(endpoint, timeout=timeout)
#         resp.raise_for_status()
#         data  = resp.json()
#         route = data["routes"][0]

#         distance_km  = round(route["distance"] / 1_000, 2)
#         duration_min = round(route["duration"] / 60,    2)

#         return distance_km, duration_min

#     except (KeyError, IndexError) as exc:
#         raise RuntimeError(
#             f"OSRM — réponse inattendue pour "
#             f"({lat1},{lon1}) → ({lat2},{lon2}) : {exc}"
#         ) from exc
#     except requests.RequestException as exc:
#         raise RuntimeError(f"OSRM — erreur réseau : {exc}") from exc
import requests

def get_osrm_distance(lat1, lon1, lat2, lon2):
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status() 
        data = res.json()
        
        route = data["routes"][0]
        distance_km = round(route["distance"] / 1000, 2)
        duration_min = round(route["duration"] / 60, 2)
        
        return distance_km, duration_min
    
    except (requests.RequestException, KeyError, IndexError) as e:
        print(f"⚠️ Erreur OSRM: {e}")
        return None, None 
# point1 = (36.848097, 10.217551)
# point2 = (36.420177, 10.553902)

# distance, duration = get_osrm_distance(*point1, *point2)

# if distance is not None:
#     print(f"Distance réelle: {distance:.2f} km")
#     print(f"Durée: {duration:.2f} min")
# else:
#     print("Impossible de calculer la distance ou la durée.")