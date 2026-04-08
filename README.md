# 🚕 MOVIROO — Pricing Engine ML

Moteur de tarification dynamique pour **58 villes tunisiennes**.  
Combine **XGBoost + LightGBM** (ensemble) avec une formule de règles métier transparente.

---


---

## 🧮 Formule de prix

```
prix_final = (base_fare + distance_km × 0.45 + duree_min × 0.08)
             × surge_multiplier
```

### Table des multiplicateurs

| Facteur         | Valeurs                          | Multiplicateur          |
|-----------------|----------------------------------|-------------------------|
| **Trafic**      | 1=faible / 2=modéré / 3=élevé   | ×1.0 / ×1.2 / ×1.5     |
| **Météo**       | 1=clair / 2=pluie / 3=tempête / 4=sirocco | ×1.0 / ×1.1 / ×1.3 / ×1.1 |
| **Demande**     | normal / rush / surge            | ×1.0 / ×1.25 / ×1.6    |
| **Nuit**        | avant 6h ou après 20h            | ×1.2                    |
| **Voiture**     | economy / comfort / van / premium | ×0.75 → ×1.6           |
| **Vendredi**    | Jumu'ah 11h–13h                  | ×1.4                    |
| **Ramadan**     | Iftar / Tarawih / Suhoor         | ×2.1 / ×1.3 / ×1.15    |
| **Beach surge** | matin / après-midi / coucher     | ×1.25 / ×1.30 / ×1.35  |
| **Zone**        | capitale / balnéaire / intérieure / sud | ×1.15 / ×1.10 / ×1.0 / ×0.95 |

### Exemple concret — Tunis Centre → Ariana, Vendredi Iftar 🌙

```
raw       = 2.50 + (8.5 × 0.45) + (22 × 0.08) = 7.085 TND
surge     = 1.5 × 1.0 × 1.6 × 1.2 × 1.0 × 1.4 × 2.1 × 1.0 × 1.15
          = 9.22 (!!)   → clamped à 3.5 (plafond sécurité ML)
final     = 7.085 × 3.5 = 24.80 TND (comfort)
```

---

## 🤖 Modèle ML

### Features d'entrée (32 features après engineering)

| Feature              | Type   | Description                       |
|----------------------|--------|-----------------------------------|
| `distance_km`        | float  | Distance du trajet                |
| `duration_min`       | float  | Durée estimée                     |
| `trafic_niveau`      | int    | 1 / 2 / 3                        |
| `weather_code`       | int    | 1–4                               |
| `hour_sin/cos`       | float  | Heure cyclique (sin/cos)          |
| `day_sin/cos`        | float  | Jour cyclique                     |
| `is_night`           | bool   | Créneau nocturne                  |
| `is_ramadan_slot`    | bool   | Slot Ramadan actif                |
| `is_friday_slot`     | bool   | Vendredi Jumu'ah                  |
| `has_beach`          | bool   | Ville balnéaire                   |
| `is_beach_hour`      | bool   | Heure de pic plage                |
| `traffic_x_demand`   | float  | Interaction trafic × demande      |
| `ramadan_x_traffic`  | float  | Interaction Ramadan × trafic      |
| ... (32 total)       |        |                                   |

### Architecture ensemble

```
XGBoost  (800 arbres, lr=0.04)  → surge_xgb   ──┐
                                                    ├─ × 0.55 + × 0.45 → surge_final
LightGBM (800 arbres, lr=0.04)  → surge_lgbm  ──┘
```

---

## 🚀 Installation & Lancement

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Entraînement ML

```bash
# Depuis la racine du projet
python models/train.py --csv chemin/vers/cleaned_data.csv
```

Sorties dans `models/` :
- `xgb_model.json`
- `lgbm_model.txt`
- `scaler.pkl`
- `feature_columns.json`
- `training_report.json`

### 3. Démarrer l'API

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Swagger UI

Ouvrir : http://localhost:8000/docs

---

## 📡 Endpoints API

### `POST /api/v1/price`

Calcul complet avec détail de chaque multiplicateur.

```bash
curl -X POST http://localhost:8000/api/v1/price \
  -H "Content-Type: application/json" \
  -d '{
    "distance_km": 8.5,
    "duration_min": 22,
    "car_type": "comfort",
    "trafic_niveau": 3,
    "demande": "surge",
    "weather_code": 2,
    "hour_of_day": 18,
    "day_of_week": 4,
    "is_night": 0,
    "is_ramadan_slot": 1,
    "periode": "ramadan_iftar",
    "is_friday_slot": 0,
    "zone_type": "capitale",
    "has_beach": 0,
    "is_beach_hour": 0,
    "beach_peak_reason": "none",
    "beach_surge_applied": 0,
    "beach_surge_value": 1.0,
    "chauffeurs_actifs": 8,
    "intensite_ville": 5,
    "population": 1056247,
    "indice_congestion": 85,
    "retard_estime_min": 25,
    "vitesse_moy_kmh": 18,
    "temperature_2m": 20.0,
    "windspeed_10m": 12.0,
    "precipitation": 3.5,
    "is_school_slot": 0,
    "is_prayer_slot": 0,
    "use_ml": true
  }'
```

**Réponse exemple :**

```json
{
  "base_fare": 2.50,
  "distance_cost": 3.825,
  "duration_cost": 1.76,
  "raw_price": 8.085,
  "surge_multiplier": 3.1200,
  "final_price": 25.22,
  "currency": "TND",
  "min_applied": false,
  "mult_traffic": 1.5,
  "mult_weather": 1.1,
  "mult_demand": 1.6,
  "mult_night": 1.0,
  "mult_car": 1.0,
  "mult_friday": 1.0,
  "mult_ramadan": 2.1,
  "mult_beach": 1.0,
  "mult_zone": 1.15,
  "ml_used": true,
  "ml_surge_xgb": 3.1145,
  "ml_surge_lgbm": 3.1270,
  "ml_confidence": 0.9960,
  "source": "ML ensemble (XGB×0.55+LGBM×0.45)",
  "computed_at": "2025-03-28T18:15:00.123456"
}
```

### `POST /api/v1/price/estimate`

Fourchette de prix pour affichage mobile.

```json
{
  "min_price": 21.44,
  "max_price": 29.00,
  "est_price": 25.22,
  "currency": "TND",
  "surge_level": "très élevé",
  "surge_mult": 3.12
}
```

### `GET /api/v1/health`

```json
{"status": "ok", "ml_loaded": true, "version": "1.0.0"}
```

### `GET /api/v1/multipliers`

Retourne toutes les tables de multiplicateurs.

---

## 🔄 Mode dégradé

Si le modèle ML n'est pas entraîné (fichiers absents dans `models/`),
l'API démarre quand même en **mode règles pures** — 100% transparent,
déterministe, sans dépendance ML.

Passer `"use_ml": false` pour forcer ce mode à tout moment.

---

## 🇹🇳 Contexte Ramadan 2025

| Slot         | Heure          | Multiplicateur |
|--------------|----------------|----------------|
| Iftar        | 17h45 – 18h15  | ×2.10          |
| Tarawih      | ~22h00         | ×1.30          |
| Suhoor       | ~02h30         | ×1.15          |
