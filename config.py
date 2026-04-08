from __future__ import annotations



BASE_FARE     = 3.50   # TND — prise en charge fixe
RATE_PER_KM   = 0.65   # TND / km
RATE_PER_MIN  = 0.30   # TND / min
MIN_FARE      = 4.00   # TND — course minimale garantie



W_XGB  = 0.55   # poids XGBoost
W_LGBM = 0.45   # poids LightGBM


MULT_TRAFFIC: dict[int, float] = {
    1: 1.00,
    2: 1.20,
    3: 1.50,
}

MULT_WEATHER: dict[int, float] = {
    1: 1.00,
    2: 2.10,
    3: 1.30,
    4: 1.10,
}

MULT_DEMAND: dict[str, float] = {
    "normal": 1.00,
    "rush":   1.25,
    "surge":  1.60,
}

MULT_NIGHT: float = 1.20

MULT_CAR: dict[str, float] = {
    "economy":     0.75,
    "standard":    0.90,
    "comfort":     1.00,
    "first_class": 1.60,
    "van":         1.30,
    "mini_bus":    1.50,
}

MULT_FRIDAY_JUMUAH: float = 1.40

MULT_RAMADAN: dict[str, float] = {
    "ramadan_iftar":         2.10,  
    "ramadan_tarawih":       1.30,   
    "ramadan_suhoor":        1.15,  
    "ramadan_last_week":     1.60,  
    "none":                  1.00,
}

MULT_BEACH: dict[str, float] = {
    "afflux_matin":   1.25,   
    "après_midi":     1.30,   
    "coucher_soleil": 1.35,   
    "none":           1.00,
}

MULT_ZONE: dict[str, float] = {
    "capitale":   1.15,
    "banlieue":   1.05,
    "balnéaire":  1.10,
    "intérieure": 1.00,
    "sud":        0.95,
}

MULT_SPECIAL_EVENT: dict[str, float] = {
    "aid_el_fitr":           2.00,
    "aid_el_adha_week":      1.80,
    "new_year_eve":          1.90,
    "new_year_days":         1.40,
    "none":                  1.00,
}

ESTIMATED_WEATHER_BY_SEASON: dict[str, dict] = {
    "été": {
        "temperature_2m":  32.0,
        "windspeed_10m":   12.0,
        "precipitation":    0.0,
        "rain":             0.0,
        "weather_code":     1,      
        "weather_label":   "clair",
        "weather_mult":    1.00,
        "weathercode_raw":  0,
        "visibility":   10_000.0,
    },
    "printemps": {
        "temperature_2m":  20.0,
        "windspeed_10m":   14.0,
        "precipitation":    5.0,
        "rain":             5.0,
        "weather_code":     2,     
        "weather_label":   "pluie",
        "weather_mult":    1.10,
        "weathercode_raw":  61,
        "visibility":    8_000.0,
    },
    "automne": {
        "temperature_2m":  18.0,
        "windspeed_10m":   15.0,
        "precipitation":    8.0,
        "rain":             8.0,
        "weather_code":     2,
        "weather_label":   "pluie",
        "weather_mult":    1.10,
        "weathercode_raw":  61,
        "visibility":    7_000.0,
    },
    "hiver": {
        "temperature_2m":  11.0,
        "windspeed_10m":   18.0,
        "precipitation":   15.0,
        "rain":            15.0,
        "weather_code":     2,
        "weather_label":   "pluie",
        "weather_mult":    1.10,
        "weathercode_raw":  63,
        "visibility":    6_000.0,
    },
}


WEATHER_LABELS: dict[int, str] = {
    1: "clair",
    2: "pluie",
    3: "tempête",
    4: "sirocco",
}


ZONE_MAP: dict[str, int] = {
    "capitale":   0,
    "banlieue":   1,
    "balnéaire":  2,
    "intérieure": 3,
    "sud":        4,
}

DEMAND_MAP: dict[str, int] = {
    "normal": 0,
    "rush":   1,
    "surge":  2,
}

CAR_MAP: dict[str, int] = {
    "economy":     1,
    "standard":    2,
    "comfort":     3,
    "van":         4,
    "mini_bus":    5,
    "first_class": 6,
}

PERIODE_MAP: dict[str, int] = {
    "nuit_calme":             0,
    "circulation_normale":    1,
    "matin_normal":           1,
    "normal":                 1,
    "rush_matin_peak":        2,
    "pause_dejeuner":         3,
    "rush_soir":              4,
    "sortie_mosquee_jumua":   5,
    "ramadan_iftar":          6,
    "ramadan_tarawih":        7,
    "ramadan_suhoor":         8,
    "ramadan_last_week":      9,
    "aid_el_fitr":           10,
    "aid_el_adha_week":      11,
    "new_year_eve":          12,
    "new_year_days":         13,
}

BEACH_REASON_MAP: dict[str, int] = {
    "":               0,
    "none":           0,
    "afflux_matin":   1,
    "après_midi":     2,
    "coucher_soleil": 3,
}


RAMADAN_TABLE: dict[int, tuple[str, str]] = {
   
    2026: ("2026-02-18", "2026-03-19"),
    2027: ("2027-02-08", "2027-03-09"),
    2028: ("2028-01-28", "2028-02-25"),
    2029: ("2029-01-16", "2029-02-13"),
    2030: ("2030-01-06", "2030-02-03"),
}

AID_ADHA_TABLE: dict[int, tuple[str, str]] = {
    2023: ("2023-06-27", "2023-07-04"),
    2024: ("2024-06-16", "2024-06-23"),
    2025: ("2025-06-06", "2025-06-13"),
    2026: ("2026-05-27", "2026-06-03"),
    2027: ("2027-05-17", "2027-05-24"),
    2028: ("2028-05-05", "2028-05-12"),
    2029: ("2029-04-24", "2029-05-01"),
    2030: ("2030-04-14", "2030-04-21"),
}


XGB_PARAMS: dict = {
    "objective":        "reg:squarederror",
    "n_estimators":     800,
    "max_depth":        7,
    "learning_rate":    0.04,
    "subsample":        0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

LGBM_PARAMS: dict = {
    "objective":         "regression",
    "n_estimators":      800,
    "max_depth":         7,
    "learning_rate":     0.04,
    "num_leaves":        63,
    "subsample":         0.80,
    "colsample_bytree":  0.80,
    "min_child_samples": 10,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":          -1,
}

TARGET_COL   = "surge_multiplier"
RANDOM_STATE = 42