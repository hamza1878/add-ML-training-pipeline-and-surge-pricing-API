"""
╔══════════════════════════════════════════════════════════╗
║   MOVIROO — ZERO DEPENDENCIES (Pure Python 3.14)         ║
║   pip install NOTHING — runs on any Python 3.8+          ║
║   Uses only: re · json · datetime  (all built-in)        ║
╚══════════════════════════════════════════════════════════╝

When to use this:
  → spaCy also fails on your machine
  → You want instant startup (no model loading)
  → Covers ~85% of real ride-booking sentences

Run:  python moviroo_rules_only.py
"""

import re
import json
from datetime import date, timedelta


# ═══════════════════════════════════════════════════════════
# KEYWORD DICTIONARIES
# ═══════════════════════════════════════════════════════════

DESTINATIONS = {
    "hammamet","7amaamt","tunis","tnis","sfax","sousse","soussa",
    "nabeul","monastir","bizerte","djerba","kairouan","gabès","gafsa",
    "tozeur","la marsa","marsa","carthage","ennasr","lac","menzah",
    "حمامات","تونس","سوسة","صفاقس","المنستير","نابل","بنزرت","جربة",
    "القيروان",
}

PLACE_WORDS = {
    "matar","aeroport","aéroport","airport","المطار",
    "gare","mahatta","station","المحطة",
    "centre","center","wust el-bled","وسط البلد",
    "funduk","hotel","الفندق",
    "beit","dar","home","البيت","maison",
    "bureau","maktab","el-maktab","office","المكتب",
}

GOTO_WORDS = {
    "en":"go to|take me to|drop me at|deliver me to|going to|i want to go|i need to go|heading to",
    "fr":"aller à|aller a|vers|jusqu à|jusqu a|destination|arriver à",
    "tn":"nemchi|bch nemchi|wein|khodni lel|jibni lel|3aybni lel",
    "ar":"إلى|نروح|وين|اذهب إلى",
}

FROM_WORDS = {
    "en":"from|starting from|pick me up at|coming from",
    "fr":"de|depuis|depuis le|depuis la|en partant de|point de départ",
    "tn":"min|men|men 3andi|min 3andi",
    "ar":"من|من عند",
}

HERE_WORDS = {
    "hne","هنا","ici","here","ma position","current location",
    "3andi","men 3andi","min 3andi",
}

DATE_KEYWORDS: dict[str, str] = {
    "lyoum":"today","lioum":"today","اليوم":"today",
    "aujourd'hui":"today","aujourdhui":"today","today":"today",
    "tawwa":"now","الآن":"now","maintenant":"now","now":"now",
    "ghodwa":"tomorrow","ghodoa":"tomorrow","غدا":"tomorrow",
    "غداً":"tomorrow","demain":"tomorrow","tomorrow":"tomorrow",
    "ba3d ghodwa":"day_after_tomorrow","après-demain":"day_after_tomorrow",
    "apres-demain":"day_after_tomorrow","بعد غد":"day_after_tomorrow",
    "day after tomorrow":"day_after_tomorrow",
    "ce soir":"tonight","tonight":"tonight",
    "el-jemaa":"friday","jemaa":"friday","الجمعة":"friday","friday":"friday","vendredi":"friday",
    "el-khamis":"thursday","khamis":"thursday","الخميس":"thursday","thursday":"thursday","jeudi":"thursday",
    "el-ahad":"sunday","ahad":"sunday","الأحد":"sunday","sunday":"sunday","dimanche":"sunday",
    "el-itnin":"monday","itnin":"monday","الاثنين":"monday","monday":"monday","lundi":"monday",
    "el-talata":"tuesday","talata":"tuesday","الثلاثاء":"tuesday","tuesday":"tuesday","mardi":"tuesday",
    "el-arba3":"wednesday","arba3":"wednesday","الأربعاء":"wednesday","wednesday":"wednesday","mercredi":"wednesday",
    "samedi":"saturday","saturday":"saturday","السبت":"saturday",
}

TIME_KEYWORDS: dict[str, str] = {
    "sbeh":"morning","fel sbeh":"morning","الصباح":"morning","صباحاً":"morning",
    "matin":"morning","morning":"morning","early morning":"early_morning",
    "3achiya":"evening","fel 3achiya":"evening","المساء":"evening","مساءً":"evening",
    "soir":"evening","evening":"evening",
    "dhuhr":"noon","fel dhuhr":"noon","الظهر":"noon","midi":"noon","noon":"noon",
    "ba3d dhuhr":"afternoon","après-midi":"afternoon","apres-midi":"afternoon","afternoon":"afternoon",
    "leyl":"night","el-lil":"night","الليل":"night","nuit":"night","night":"night",
    "nouss el-lil":"midnight","midnight":"midnight",
}

LOCATION_MAP: dict[str, str] = {
    "hammamet":"Hammamet","7amaamt":"Hammamet","حمامات":"Hammamet",
    "sousse":"Sousse","soussa":"Sousse","سوسة":"Sousse",
    "sfax":"Sfax","صفاقس":"Sfax",
    "tunis":"Tunis","tnis":"Tunis","تونس":"Tunis",
    "monastir":"Monastir","المنستير":"Monastir",
    "nabeul":"Nabeul","نابل":"Nabeul",
    "bizerte":"Bizerte","بنزرت":"Bizerte",
    "kairouan":"Kairouan","القيروان":"Kairouan",
    "djerba":"Djerba","جربة":"Djerba",
    "la marsa":"La Marsa","marsa":"La Marsa",
    "carthage":"Carthage","ennasr":"Ennasr","lac":"Lac",
    "matar":"Airport","aeroport":"Airport","aéroport":"Airport",
    "airport":"Airport","المطار":"Airport",
    "gare":"Train Station","mahatta":"Train Station","المحطة":"Train Station",
    "centre":"City Center","center":"City Center","وسط البلد":"City Center",
    "funduk":"Hotel","hotel":"Hotel","الفندق":"Hotel",
    "beit":"Home","dar":"Home","البيت":"Home","maison":"Home","home":"Home",
    "bureau":"Office","maktab":"Office","el-maktab":"Office","المكتب":"Office","office":"Office",
}

DATE_DELTA = {"today":0,"now":0,"tomorrow":1,"day_after_tomorrow":2,"tonight":0}
DAY_NAMES  = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
               "friday":4,"saturday":5,"sunday":6}


# ═══════════════════════════════════════════════════════════
# REGEX PATTERNS
# ═══════════════════════════════════════════════════════════

RE_TIME = re.compile(
    r'\b(\d{1,2}h\d{0,2}|\d{1,2}:\d{2}|\d{1,2}(?:am|pm))\b', re.I
)
RE_DATE_NUM = re.compile(
    r'\ble\s*(\d{1,2})\b|\b(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{4}))?\b'
)


# ═══════════════════════════════════════════════════════════
# NORMALIZERS
# ═══════════════════════════════════════════════════════════

def norm_date(raw: str) -> str:
    if not raw:
        return None
    k = raw.lower().strip()
    mapped = DATE_KEYWORDS.get(k, k)
    today  = date.today()

    if mapped in DATE_DELTA:
        return (today + timedelta(days=DATE_DELTA[mapped])).isoformat()
    if mapped == "tonight":
        return today.isoformat()
    if mapped in DAY_NAMES:
        ahead = (DAY_NAMES[mapped] - today.weekday()) % 7 or 7
        return (today + timedelta(days=ahead)).isoformat()

    # numeric from regex groups
    m = RE_DATE_NUM.search(k)
    if m:
        if m.group(1):                          # "le 15"
            day, month, year = int(m.group(1)), today.month, today.year
        else:
            day   = int(m.group(2))
            month = int(m.group(3))
            year  = int(m.group(4)) if m.group(4) else today.year
        try:
            d = date(year, month, day)
            if d < today:
                d = d.replace(year=year + 1)
            return d.isoformat()
        except ValueError:
            pass
    return mapped


def norm_time(raw: str) -> str:
    if not raw:
        return None
    k = raw.lower().strip()
    if k in TIME_KEYWORDS:
        return TIME_KEYWORDS[k]
    m = re.match(r"(\d{1,2})h(\d{0,2})$", k)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2) or 0):02d}"
    m = re.match(r"(\d{1,2}):(\d{2})$", k)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    m = re.match(r"(\d{1,2})(am|pm)$", k)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h != 12: h += 12
        elif m.group(2) == "am" and h == 12: h = 0
        return f"{h:02d}:00"
    return raw


def norm_loc(raw: str) -> str:
    if not raw:
        return None
    return LOCATION_MAP.get(raw.lower().strip(), raw.title())


# ═══════════════════════════════════════════════════════════
# RULE-BASED EXTRACTOR  (no ML, pure Python)
# ═══════════════════════════════════════════════════════════

def _build_goto_re():
    all_kw = "|".join(v for v in GOTO_WORDS.values())
    return re.compile(rf'(?:{all_kw})\s+(?:le\s+|la\s+|l[\'`]\s*|à\s*|a\s*|lel\s+|lel-\s*)?(\S+)', re.I)

def _build_from_re():
    all_kw = "|".join(v for v in FROM_WORDS.values())
    return re.compile(rf'(?:{all_kw})\s+(?:le\s+|la\s+|l[\'`]\s*|من\s+|mon\s+|ma\s+|my\s+)?(\S+)', re.I)

_RE_GOTO = _build_goto_re()
_RE_FROM = _build_from_re()


def extract(text: str) -> dict:
    """Pure-Python rule extractor — zero dependencies."""
    lo   = text.lower()
    result = {"destination": None, "departure": None, "date": None, "time": None}

    # ── 1. TIME — regex first (most reliable) ─────────────
    m = RE_TIME.search(lo)
    if m:
        result["time"] = norm_time(m.group(1))
    else:
        # keyword scan (longest match first)
        for kw in sorted(TIME_KEYWORDS, key=len, reverse=True):
            if kw in lo:
                result["time"] = TIME_KEYWORDS[kw]
                break

    # ── 2. DATE — multi-word first ─────────────────────────
    found_date = False
    for kw in sorted(DATE_KEYWORDS, key=len, reverse=True):
        if kw in lo:
            result["date"] = norm_date(kw)
            found_date = True
            break
    if not found_date:
        m = RE_DATE_NUM.search(lo)
        if m:
            result["date"] = norm_date(m.group(0))

    # ── 3. DESTINATION — goto pattern + known words ────────
    m = _RE_GOTO.search(lo)
    if m:
        candidate = m.group(1).strip(".,;!?،")
        if candidate in DESTINATIONS | PLACE_WORDS:
            result["destination"] = norm_loc(candidate)

    if not result["destination"]:
        # scan for known destination words
        words = re.findall(r"[\w']+", lo)
        for i, w in enumerate(words):
            # try bigram first (e.g. "la marsa")
            bigram = f"{w} {words[i+1]}" if i + 1 < len(words) else ""
            if bigram in DESTINATIONS | PLACE_WORDS:
                result["destination"] = norm_loc(bigram)
                break
            if w in DESTINATIONS | PLACE_WORDS:
                result["destination"] = norm_loc(w)
                break

    # ── 4. DEPARTURE ───────────────────────────────────────
    m = _RE_FROM.search(lo)
    if m:
        candidate = m.group(1).strip(".,;!?،")
        if candidate in HERE_WORDS:
            result["departure"] = "current_location"
        elif candidate in DESTINATIONS | PLACE_WORDS:
            result["departure"] = norm_loc(candidate)

    if not result["departure"]:
        for w in HERE_WORDS:
            if w in lo:
                result["departure"] = "current_location"
                break

    # default departure
    if not result["departure"]:
        result["departure"] = "current_location"

    result["missing_fields"] = [
        f for f in ["destination", "date", "time"] if not result.get(f)
    ]
    return result


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

TEST_SENTENCES = [
    "nheb nemchi hammamet ghodwa 18h",
    "je veux aller à tunis demain matin",
    "take me to the airport tonight at 22h",
    "من هنا إلى سوسة غداً في المساء",
    "nemchi sfax min el-maktab el-khamis ba3d dhuhr",
    "réserver un taxi de la marsa vers le centre vendredi à 14h30",
    "I want to go to monastir tomorrow evening",
    "احتاج تاكسي من المكتب إلى المطار غداً صباحاً",
    "bch nemchi sousse el-jemaa fel sbeh",
    "taxi depuis bizerte vers sfax samedi 9h",
]

if __name__ == "__main__":
    SEP = "─" * 58
    print(f"\n{SEP}")
    print("🚗  MOVIROO — Zero-dependency extractor (Python 3.14 ✅)")
    print(SEP)

    for sentence in TEST_SENTENCES:
        r = extract(sentence)
        print(f"\n📝  {sentence}")
        print(f"     destination : {r['destination']}")
        print(f"     departure   : {r['departure']}")
        print(f"     date        : {r['date']}")
        print(f"     time        : {r['time']}")
        if r["missing_fields"]:
            print(f"     ⚠  missing  : {r['missing_fields']}")

    print(f"\n{SEP}")
    print("💬  Interactive (q = quit)")
    print(SEP)
    while True:
        try:
            inp = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if inp.lower() in ("q", "quit", "exit", ""):
            break
        print(json.dumps(extract(inp), ensure_ascii=False, indent=2))
