

from __future__ import annotations

from datetime import datetime

import pandas as pd

from config import RAMADAN_TABLE


def get_season(dt: datetime) -> str:
 
    m = dt.month
    if   m in (6, 7, 8, 9):  return "été"
    elif m in (3, 4, 5):      return "printemps"
    elif m in (10, 11):       return "automne"
    else:                     return "hiver"



def get_ramadan_period(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
   
    if year in RAMADAN_TABLE:
        s, e = RAMADAN_TABLE[year]
        return pd.Timestamp(s), pd.Timestamp(e)

    base  = pd.Timestamp("2025-03-01")
    delta = (year - 2025) * (-10.9)
    start = base + pd.Timedelta(days=round(delta))
    return start, start + pd.Timedelta(days=29)


def _is_in_ramadan(dt: datetime) -> bool:
    start, end = get_ramadan_period(dt.year)
    return start <= pd.Timestamp(dt) <= end



def compute_time_flags(dt: datetime) -> dict:
   
    h  = dt.hour
    m  = dt.minute
    wd = dt.weekday()   # 0=lundi … 6=dimanche

    in_ram     = _is_in_ramadan(dt)
    is_iftar   = in_ram and ((h == 17 and m >= 45) or (h == 18 and m <= 15))
    is_tarawih = in_ram and h == 22
    is_suhoor  = in_ram and (h == 2 and m >= 20)
    is_ramadan = int(is_iftar or is_tarawih or is_suhoor)

    is_friday  = int(wd == 4 and 11 <= h <= 13)

    is_school  = int(
        (h == 12 and 10 <= m <= 20)
        or (h == 17 and 10 <= m <= 20)
    )

    is_prayer  = int(h in (4, 12) and m <= 15)

    if   is_iftar:              periode = "ramadan_iftar"
    elif is_tarawih:            periode = "ramadan_tarawih"
    elif is_suhoor:             periode = "ramadan_suhoor"
    elif is_friday:             periode = "sortie_mosquee_jumua"
    elif h in (7, 8):           periode = "rush_matin_peak"
    elif h in (17, 18):        periode = "rush_soir"
    elif h in (9, 10, 11):     periode = "matin_normal"
    elif h in (12, 13):        periode = "pause_dejeuner"
    elif h < 6 or h >= 20:     periode = "nuit_calme"
    else:                       periode = "circulation_normale"

    return {
        "heure_int":       h,
        "minute":          m,
        "jour_semaine":    wd,
        "is_ramadan_slot": is_ramadan,
        "is_friday_slot":  is_friday,
        "is_school_slot":  is_school,
        "is_prayer_slot":  is_prayer,
        "periode":         periode,
        "season":          get_season(dt),
    }



_BEACH_MULT: dict[str, dict[str, float]] = {
    "été":       {"afflux_matin": 1.25, "après_midi": 1.35, "coucher_soleil": 1.40},
    "printemps": {"afflux_matin": 1.15, "après_midi": 1.20, "coucher_soleil": 1.25},
    "automne":   {"afflux_matin": 1.10, "après_midi": 1.15, "coucher_soleil": 1.20},
    "hiver":     {"afflux_matin": 1.00, "après_midi": 1.00, "coucher_soleil": 1.00},
}

_EMPTY_BEACH: dict = {
    "is_beach_hour":       0,
    "beach_peak_reason":   "none",
    "beach_surge_applied": 0,
    "beach_surge_value":   1.0,
}


def compute_beach_flags(has_beach: int, dt: datetime) -> dict:
   
    if not has_beach:
        return _EMPTY_BEACH.copy()

    season = get_season(dt)
    if season == "hiver":
        return _EMPTY_BEACH.copy()

    h     = dt.hour
    mults = _BEACH_MULT[season]

    if   9  <= h <= 11:  reason = "afflux_matin"
    elif 14 <= h <= 16:  reason = "après_midi"
    elif 19 <= h <= 20:  reason = "coucher_soleil"
    else:
        return _EMPTY_BEACH.copy()

    surge_val = mults[reason]
    if surge_val <= 1.0:
        return _EMPTY_BEACH.copy()

    return {
        "is_beach_hour":       1,
        "beach_peak_reason":   reason,
        "beach_surge_applied": 1,
        "beach_surge_value":   surge_val,
    }
