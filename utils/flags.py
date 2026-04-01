from __future__ import annotations

from datetime import datetime

import pandas as pd

from config import RAMADAN_TABLE, AID_ADHA_TABLE


# ══════════════════════════════════════════════════════════════════
# SAISON
# ══════════════════════════════════════════════════════════════════

def get_season(dt: datetime) -> str:
    m = dt.month
    if   m in (6, 7, 8, 9):  return "été"
    elif m in (3, 4, 5):      return "printemps"
    elif m in (10, 11):       return "automne"
    else:                     return "hiver"


# ══════════════════════════════════════════════════════════════════
# RAMADAN — bornes + dernière semaine
# ══════════════════════════════════════════════════════════════════

def get_ramadan_period(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    if year in RAMADAN_TABLE:
        s, e = RAMADAN_TABLE[year]
        return pd.Timestamp(s), pd.Timestamp(e)
    # Approximation : Ramadan recule d'~11 jours par an
    base  = pd.Timestamp("2025-03-01")
    delta = (year - 2025) * (-10.9)
    start = base + pd.Timedelta(days=round(delta))
    return start, start + pd.Timedelta(days=29)


def _is_in_ramadan(dt: datetime) -> bool:
    start, end = get_ramadan_period(dt.year)
    return start <= pd.Timestamp(dt) <= end


def _is_ramadan_last_week(dt: datetime) -> bool:
    """Retourne True si on est dans les 7 derniers jours du Ramadan."""
    start, end = get_ramadan_period(dt.year)
    last_week_start = end - pd.Timedelta(days=6)
    ts = pd.Timestamp(dt)
    return last_week_start <= ts <= end


# ══════════════════════════════════════════════════════════════════
# AÏD EL-FITR  (3 jours après fin Ramadan)
# ══════════════════════════════════════════════════════════════════

def _get_aid_fitr_period(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    _, ram_end = get_ramadan_period(year)
    aid_start  = ram_end + pd.Timedelta(days=1)
    aid_end    = aid_start + pd.Timedelta(days=2)   # 3 jours
    return aid_start, aid_end


def _is_aid_el_fitr(dt: datetime) -> bool:
    start, end = _get_aid_fitr_period(dt.year)
    return start <= pd.Timestamp(dt) <= end


# ══════════════════════════════════════════════════════════════════
# AÏD EL-ADHA  (semaine complète)
# ══════════════════════════════════════════════════════════════════

def _get_aid_adha_period(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    if year in AID_ADHA_TABLE:
        s, e = AID_ADHA_TABLE[year]
        return pd.Timestamp(s), pd.Timestamp(e)
    # Approximation : Aïd el-Adha recule d'~11 jours par an
    base  = pd.Timestamp("2025-06-06")
    delta = (year - 2025) * (-10.9)
    start = base + pd.Timedelta(days=round(delta))
    return start, start + pd.Timedelta(days=7)


def _is_aid_adha_week(dt: datetime) -> bool:
    start, end = _get_aid_adha_period(dt.year)
    return start <= pd.Timestamp(dt) <= end


# ══════════════════════════════════════════════════════════════════
# NOUVEL AN
# ══════════════════════════════════════════════════════════════════

def _is_new_year_eve(dt: datetime) -> bool:
    """31 décembre à partir de 20h."""
    return dt.month == 12 and dt.day == 31 and dt.hour >= 20


def _is_new_year_days(dt: datetime) -> bool:
    """2 et 3 janvier."""
    return dt.month == 1 and dt.day in (2, 3)


# ══════════════════════════════════════════════════════════════════
# FLAGS PRINCIPAUX
# ══════════════════════════════════════════════════════════════════

def compute_time_flags(dt: datetime) -> dict:
    """
    Calcule tous les flags temporels et culturels pour une datetime.

    Retourne un dict contenant :
        heure_int, minute, jour_semaine
        is_ramadan_slot, is_ramadan_last_week
        is_aid_el_fitr, is_aid_adha_week
        is_new_year_eve, is_new_year_days
        is_friday_slot, is_school_slot, is_prayer_slot
        special_event   str  — identifiant de l'événement prioritaire
        periode         str  — créneau horaire détaillé
        season          str
    """
    h  = dt.hour
    m  = dt.minute
    wd = dt.weekday()   # 0=lundi … 6=dimanche

    # ── Ramadan ───────────────────────────────────────────────────
    in_ram        = _is_in_ramadan(dt)
    in_ram_last_w = _is_ramadan_last_week(dt)
    is_iftar      = in_ram and ((h == 17 and m >= 45) or (h == 18 and m <= 15))
    is_tarawih    = in_ram and h == 22
    is_suhoor     = in_ram and (h == 2 and m >= 20)
    is_ramadan    = int(is_iftar or is_tarawih or is_suhoor)

    # ── Aïd ───────────────────────────────────────────────────────
    is_aid_fitr   = int(_is_aid_el_fitr(dt))
    is_aid_adha   = int(_is_aid_adha_week(dt))

    # ── Nouvel An ─────────────────────────────────────────────────
    is_nye        = int(_is_new_year_eve(dt))
    is_nyd        = int(_is_new_year_days(dt))

    # ── Autres flags ──────────────────────────────────────────────
    is_friday     = int(wd == 4 and 11 <= h <= 13)
    is_school     = int(
        (h == 12 and 10 <= m <= 20)
        or (h == 17 and 10 <= m <= 20)
    )
    is_prayer     = int(h in (4, 12) and m <= 15)

    # ── Événement spécial prioritaire (pour pricing) ──────────────
    if is_aid_fitr:
        special_event = "aid_el_fitr"
    elif is_aid_adha:
        special_event = "aid_el_adha_week"
    elif is_nye:
        special_event = "new_year_eve"
    elif is_nyd:
        special_event = "new_year_days"
    else:
        special_event = "none"

    # ── Période détaillée ─────────────────────────────────────────
    if is_aid_fitr:
        periode = "aid_el_fitr"
    elif is_aid_adha:
        periode = "aid_el_adha_week"
    elif is_nye:
        periode = "new_year_eve"
    elif is_nyd:
        periode = "new_year_days"
    elif is_iftar:
        periode = "ramadan_iftar"
    elif is_tarawih:
        periode = "ramadan_tarawih"
    elif is_suhoor:
        periode = "ramadan_suhoor"
    elif in_ram_last_w and in_ram:
        periode = "ramadan_last_week"
    elif is_friday:
        periode = "sortie_mosquee_jumua"
    elif h in (7, 8):
        periode = "rush_matin_peak"
    elif h in (17, 18):
        periode = "rush_soir"
    elif h in (9, 10, 11):
        periode = "matin_normal"
    elif h in (12, 13):
        periode = "pause_dejeuner"
    elif h < 6 or h >= 20:
        periode = "nuit_calme"
    else:
        periode = "circulation_normale"

    return {
        "heure_int":              h,
        "minute":                 m,
        "jour_semaine":           wd,
        # Flags Ramadan
        "is_ramadan_slot":        is_ramadan,
        "is_ramadan_last_week":   int(in_ram_last_w and in_ram),
        # Flags Aïd
        "is_aid_el_fitr":         is_aid_fitr,
        "is_aid_adha_week":       is_aid_adha,
        # Flags Nouvel An
        "is_new_year_eve":        is_nye,
        "is_new_year_days":       is_nyd,
        # Autres flags culturels / scolaires
        "is_friday_slot":         is_friday,
        "is_school_slot":         is_school,
        "is_prayer_slot":         is_prayer,
        # Synthèse
        "special_event":          special_event,
        "periode":                periode,
        "season":                 get_season(dt),
    }


# ══════════════════════════════════════════════════════════════════
# FLAGS BEACH  (saison-aware)
# ══════════════════════════════════════════════════════════════════

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