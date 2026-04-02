"""
╔══════════════════════════════════════════════════════════╗
║     MOVIROO — NER  ✅ Python 3.14 Compatible             ║
║     NO transformers · NO tokenizers · NO Rust            ║
║     ONLY: spaCy + pure Python rules                      ║
╚══════════════════════════════════════════════════════════╝

Install (works on Python 3.14):
    pip install spacy
    python -m spacy download xx_ent_wiki_sm
"""

import re
import json
import random
from datetime import date, timedelta

# ── spaCy (has Python 3.14 wheels ✅) ─────────────────────
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding


# ═══════════════════════════════════════════════════════════
# 1. TRAINING DATA
# ═══════════════════════════════════════════════════════════

TRAIN_DATA = [
    # ── Tunisian ──────────────────────────────────────────
    ("nheb nemchi hammamet ghodwa 18h", {
        "entities": [
            (13, 22, "DESTINATION"),
            (23, 29, "DATE"),
            (30, 33, "TIME"),
        ]
    }),
    ("khodni lel matar tawwa", {
        "entities": [
            (11, 16, "DESTINATION"),
            (17, 22, "DATE"),
        ]
    }),
    ("bch nemchi sousse el-jemaa fel sbeh", {
        "entities": [
            (11, 17, "DESTINATION"),
            (18, 26, "DATE"),
            (31, 36, "TIME"),
        ]
    }),
    ("nemchi sfax min el-maktab el-khamis ba3d dhuhr", {
        "entities": [
            (7,  11, "DESTINATION"),
            (16, 25, "DEPARTURE"),
            (26, 35, "DATE"),
            (36, 46, "TIME"),
        ]
    }),
    ("3aybni taxi men hne lel funduk el-ahad 3achiya", {
        "entities": [
            (16, 19, "DEPARTURE"),
            (24, 30, "DESTINATION"),
            (31, 38, "DATE"),
            (39, 47, "TIME"),
        ]
    }),
    ("nheb nemchi aeroport ghodwa 8h30 min beit", {
        "entities": [
            (12, 20, "DESTINATION"),
            (21, 27, "DATE"),
            (28, 32, "TIME"),
            (37, 41, "DEPARTURE"),
        ]
    }),
    ("wein bch nemchi lyoum el-lil", {
        "entities": [
            (16, 21, "DATE"),
            (22, 28, "TIME"),
        ]
    }),
    ("khodni men dar lel matar el-itnin sbeh", {
        "entities": [
            (10, 13, "DEPARTURE"),
            (18, 23, "DESTINATION"),
            (24, 32, "DATE"),
            (33, 37, "TIME"),
        ]
    }),
    ("nemchi monastir ba3d ghodwa", {
        "entities": [
            (7,  15, "DESTINATION"),
            (16, 27, "DATE"),
        ]
    }),
    ("taxi men ennasr lel lac tawwa", {
        "entities": [
            (9,  15, "DEPARTURE"),
            (20, 23, "DESTINATION"),
            (24, 29, "DATE"),
        ]
    }),

    # ── French ────────────────────────────────────────────
    ("je veux aller à tunis demain matin", {
        "entities": [
            (17, 22, "DESTINATION"),
            (23, 29, "DATE"),
            (30, 35, "TIME"),
        ]
    }),
    ("taxi de la marsa vers le centre demain à 14h", {
        "entities": [
            (8,  16, "DEPARTURE"),
            (22, 28, "DESTINATION"),
            (29, 35, "DATE"),
            (38, 41, "TIME"),
        ]
    }),
    ("aller de bizerte à nabeul mercredi à 9h", {
        "entities": [
            (9,  16, "DEPARTURE"),
            (19, 25, "DESTINATION"),
            (26, 34, "DATE"),
            (37, 39, "TIME"),
        ]
    }),
    ("taxi depuis mon bureau jusqu à sfax vendredi soir", {
        "entities": [
            (15, 21, "DEPARTURE"),
            (30, 34, "DESTINATION"),
            (35, 43, "DATE"),
            (44, 48, "TIME"),
        ]
    }),
    ("aller à sousse ce soir à 20h", {
        "entities": [
            (8,  14, "DESTINATION"),
            (15, 22, "DATE"),
            (25, 28, "TIME"),
        ]
    }),
    ("je pars de tunis vers monastir lundi matin", {
        "entities": [
            (12, 17, "DEPARTURE"),
            (23, 31, "DESTINATION"),
            (32, 37, "DATE"),
            (38, 43, "TIME"),
        ]
    }),
    ("réserver un taxi pour demain après-midi à sfax", {
        "entities": [
            (21, 27, "DATE"),
            (28, 39, "TIME"),
            (42, 47, "DESTINATION"),
        ]
    }),

    # ── English ───────────────────────────────────────────
    ("I need a taxi from sfax to sousse tomorrow at 8h30", {
        "entities": [
            (18, 22, "DEPARTURE"),
            (26, 32, "DESTINATION"),
            (33, 41, "DATE"),
            (45, 49, "TIME"),
        ]
    }),
    ("take me to the airport tonight at 22h", {
        "entities": [
            (15, 22, "DESTINATION"),
            (23, 30, "DATE"),
            (34, 37, "TIME"),
        ]
    }),
    ("drop me at the hotel from office friday morning", {
        "entities": [
            (15, 20, "DESTINATION"),
            (26, 32, "DEPARTURE"),
            (33, 39, "DATE"),
            (40, 47, "TIME"),
        ]
    }),
    ("book a ride to monastir next monday at 7h", {
        "entities": [
            (14, 22, "DESTINATION"),
            (23, 34, "DATE"),
            (38, 40, "TIME"),
        ]
    }),
    ("I want to go to hammamet tomorrow evening", {
        "entities": [
            (16, 24, "DESTINATION"),
            (25, 33, "DATE"),
            (34, 41, "TIME"),
        ]
    }),
    ("take me from my house to the station on saturday", {
        "entities": [
            (14, 22, "DEPARTURE"),
            (27, 34, "DESTINATION"),
            (38, 46, "DATE"),
        ]
    }),

    # ── Arabic ────────────────────────────────────────────
    ("من هنا إلى سوسة غداً في المساء", {
        "entities": [
            (3,  6,  "DEPARTURE"),
            (10, 14, "DESTINATION"),
            (15, 19, "DATE"),
            (23, 29, "TIME"),
        ]
    }),
    ("أريد سيارة إلى تونس الآن", {
        "entities": [
            (14, 18, "DESTINATION"),
            (19, 23, "DATE"),
        ]
    }),
    ("من المطار إلى صفاقس يوم الجمعة صباحاً", {
        "entities": [
            (3,  9,  "DEPARTURE"),
            (13, 19, "DESTINATION"),
            (24, 31, "DATE"),
            (32, 38, "TIME"),
        ]
    }),
    ("احتاج تاكسي من المكتب إلى المطار غداً", {
        "entities": [
            (16, 22, "DEPARTURE"),
            (26, 32, "DESTINATION"),
            (33, 37, "DATE"),
        ]
    }),
]


# ═══════════════════════════════════════════════════════════
# 2. TRAIN SPACY NER  (pure Python — no Rust, no C extensions)
# ═══════════════════════════════════════════════════════════

def train_spacy_model(
    train_data: list,
    n_iter: int = 120,
    output_path: str = "moviroo_model",
    dropout: float = 0.35,
) -> object:
    """
    Train a spaCy NER model.
    Works on Python 3.14 ✅ — no Rust, no binary wheels needed.
    """

    # language-agnostic blank model (xx = multilingual)
    nlp = spacy.blank("xx")
    ner = nlp.add_pipe("ner", last=True)

    # register all entity labels
    for _, ann in train_data:
        for _, _, label in ann["entities"]:
            ner.add_label(label)

    # convert to spaCy Example objects (skip invalid spans)
    examples = []
    for text, ann in train_data:
        doc = nlp.make_doc(text)
        try:
            ex = Example.from_dict(doc, ann)
            examples.append(ex)
        except Exception as e:
            print(f"  ⚠  Skipping bad example: {text[:40]}... → {e}")

    # initialize with examples
    optimizer = nlp.initialize(lambda: examples)

    print(f"🚀 Training spaCy NER — {n_iter} iterations, {len(examples)} examples\n")

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    with nlp.disable_pipes(*other_pipes):
        for i in range(n_iter):
            random.shuffle(examples)
            losses = {}
            batches = minibatch(examples, size=compounding(2.0, 16.0, 1.001))
            for batch in batches:
                nlp.update(batch, sgd=optimizer, drop=dropout, losses=losses)

            if i % 20 == 0 or i == n_iter - 1:
                print(f"  iter {i:4d} | loss: {losses.get('ner', 0):.4f}")

    nlp.to_disk(output_path)
    print(f"\n✅  Model saved → {output_path}/\n")
    return nlp


# ═══════════════════════════════════════════════════════════
# 3. PURE-PYTHON RULE ENGINE  (handles what ML misses)
#    NO ML needed here — just dict lookups + regex
# ═══════════════════════════════════════════════════════════

# ── 3a. Normalization maps ────────────────────────────────

DATE_MAP: dict[str, str] = {
    # Tunisian
    "lyoum": "today",         "lioum": "today",
    "tawwa": "now",           "tawwali": "now",
    "ghodwa": "tomorrow",     "ghodoa": "tomorrow",
    "ba3d ghodwa": "day_after_tomorrow",
    "el-jemaa": "friday",     "jemaa": "friday",
    "el-khamis": "thursday",  "khamis": "thursday",
    "el-ahad": "sunday",      "ahad": "sunday",
    "el-itnin": "monday",     "itnin": "monday",
    "el-talata": "tuesday",   "talata": "tuesday",
    "el-arba3": "wednesday",  "arba3": "wednesday",
    "el-jom3a el-jaya": "next_friday",
    "el-jom3a jaya": "next_friday",
    # French
    "aujourd'hui": "today",   "aujourdhui": "today",
    "demain": "tomorrow",     "après-demain": "day_after_tomorrow",
    "apres-demain": "day_after_tomorrow",
    "maintenant": "now",      "ce soir": "tonight",
    "ce matin": "this_morning",
    "lundi": "monday",        "mardi": "tuesday",
    "mercredi": "wednesday",  "jeudi": "thursday",
    "vendredi": "friday",     "samedi": "saturday",
    "dimanche": "sunday",     "weekend": "weekend",
    "semaine prochaine": "next_week",
    "cette semaine": "this_week",
    # English
    "today": "today",         "tomorrow": "tomorrow",
    "day after tomorrow": "day_after_tomorrow",
    "now": "now",             "tonight": "tonight",
    "monday": "monday",       "tuesday": "tuesday",
    "wednesday": "wednesday", "thursday": "thursday",
    "friday": "friday",       "saturday": "saturday",
    "sunday": "sunday",       "next week": "next_week",
    "this week": "this_week", "weekend": "weekend",
    # Arabic
    "اليوم": "today",         "غدا": "tomorrow",
    "غداً": "tomorrow",       "بعد غد": "day_after_tomorrow",
    "الآن": "now",            "الجمعة": "friday",
    "الخميس": "thursday",     "الاثنين": "monday",
}

TIME_MAP: dict[str, str] = {
    # Tunisian
    "sbeh": "morning",        "fel sbeh": "morning",
    "3achiya": "evening",     "fel 3achiya": "evening",
    "dhuhr": "noon",          "fel dhuhr": "noon",
    "ba3d dhuhr": "afternoon","leyl": "night",
    "nouss el-lil": "midnight","el-lil": "night",
    # French
    "matin": "morning",       "soir": "evening",
    "midi": "noon",           "nuit": "night",
    "après-midi": "afternoon","apres-midi": "afternoon",
    "tôt le matin": "early_morning",
    # English
    "morning": "morning",     "evening": "evening",
    "noon": "noon",           "night": "night",
    "afternoon": "afternoon", "midnight": "midnight",
    "early morning": "early_morning",
    # Arabic
    "الصباح": "morning",      "المساء": "evening",
    "صباحاً": "morning",      "مساءً": "evening",
    "الليل": "night",         "الظهر": "noon",
}

LOCATION_ALIASES: dict[str, str] = {
    # Tunisian cities (variants → canonical)
    "7amaamt": "Hammamet",    "hamamet": "Hammamet",
    "hammamet": "Hammamet",   "حمامات": "Hammamet",
    "sousse": "Sousse",       "soussa": "Sousse",
    "سوسة": "Sousse",
    "sfax": "Sfax",           "صفاقس": "Sfax",
    "tunis": "Tunis",         "tnis": "Tunis",
    "تونس": "Tunis",
    "monastir": "Monastir",   "المنستير": "Monastir",
    "nabeul": "Nabeul",       "نابل": "Nabeul",
    "bizerte": "Bizerte",     "بنزرت": "Bizerte",
    "kairouan": "Kairouan",   "القيروان": "Kairouan",
    "djerba": "Djerba",       "جربة": "Djerba",
    "ennasr": "Ennasr",       "lac": "Lac",
    "la marsa": "La Marsa",   "marsa": "La Marsa",
    "carthage": "Carthage",
    # Special places
    "matar": "Airport",       "aeroport": "Airport",
    "aéroport": "Airport",    "airport": "Airport",
    "المطار": "Airport",
    "gare": "Train Station",  "mahatta": "Train Station",
    "المحطة": "Train Station",
    "centre": "City Center",  "center": "City Center",
    "wust el-bled": "City Center",
    "funduk": "Hotel",        "hotel": "Hotel",
    "l'hotel": "Hotel",       "الفندق": "Hotel",
    "beit": "Home",           "dar": "Home",
    "البيت": "Home",          "home": "Home",
    "bureau": "Office",       "maktab": "Office",
    "el-maktab": "Office",    "المكتب": "Office",
    "office": "Office",
}

# date keyword → relative days
DATE_DELTA: dict[str, int] = {
    "today": 0, "now": 0,
    "tomorrow": 1,
    "day_after_tomorrow": 2,
}

DAY_NAMES: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}


# ── 3b. Normalization functions ───────────────────────────

def normalize_date(raw: str | None) -> str | None:
    """Convert raw date string → ISO date or named constant."""
    if not raw:
        return None
    key = raw.lower().strip()
    mapped = DATE_MAP.get(key, key)

    today = date.today()

    # relative
    if mapped in DATE_DELTA:
        return (today + timedelta(days=DATE_DELTA[mapped])).isoformat()

    # tonight = today
    if mapped == "tonight":
        return today.isoformat()

    # named weekday → next occurrence
    if mapped in DAY_NAMES:
        target_wd = DAY_NAMES[mapped]
        days_ahead = (target_wd - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7          # "friday" always means next friday if today is friday
        return (today + timedelta(days=days_ahead)).isoformat()

    # numeric: "le 15" / "15/04" / "15/04/2025"
    m = re.match(r"le\s*(\d{1,2})$", key)
    if m:
        day = int(m.group(1))
        d = today.replace(day=day)
        if d < today:
            # next month
            if today.month == 12:
                d = d.replace(year=today.year + 1, month=1)
            else:
                d = d.replace(month=today.month + 1)
        return d.isoformat()

    m = re.match(r"(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{4}))?$", key)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass

    return mapped   # return as-is if we can't normalize


def normalize_time(raw: str | None) -> str | None:
    """Convert raw time string → HH:MM or named period."""
    if not raw:
        return None
    key = raw.lower().strip()

    if key in TIME_MAP:
        return TIME_MAP[key]

    # 18h / 8h30 / 18h00
    m = re.match(r"(\d{1,2})h(\d{0,2})$", key)
    if m:
        h  = int(m.group(1))
        mn = int(m.group(2)) if m.group(2) else 0
        return f"{h:02d}:{mn:02d}"

    # 14:00 / 8:30
    m = re.match(r"(\d{1,2}):(\d{2})$", key)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"

    # 7am / 9pm
    m = re.match(r"(\d{1,2})(am|pm)$", key)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h != 12:
            h += 12
        elif m.group(2) == "am" and h == 12:
            h = 0
        return f"{h:02d}:00"

    return raw


def normalize_location(raw: str | None) -> str | None:
    """Normalize location alias → canonical name."""
    if not raw:
        return None
    key = raw.lower().strip()
    return LOCATION_ALIASES.get(key, raw.title())


# ── 3c. Rule-based fallback (regex, for when NER misses) ──

# Patterns that NER might miss — applied AFTER NER
RULE_PATTERNS = [
    # times: 18h, 8h30, 14:00, 7am
    (re.compile(r"\b(\d{1,2}h\d{0,2}|\d{1,2}:\d{2}|\d{1,2}(?:am|pm))\b", re.I), "TIME"),
    # explicit date numbers: le 15 / 15/04
    (re.compile(r"\ble\s*\d{1,2}\b|\b\d{1,2}[/\-]\d{1,2}\b", re.I), "DATE"),
]

def apply_rules(text: str, result: dict) -> dict:
    """Fill any still-missing fields using regex patterns."""
    for pattern, field in RULE_PATTERNS:
        if result.get(field.lower()):
            continue        # already found by NER
        m = pattern.search(text)
        if m:
            result[field.lower()] = m.group(0)
    return result


# ── 3d. Full postprocessing ───────────────────────────────

def postprocess(raw: dict, text: str = "") -> dict:
    """
    1. Apply rule fallbacks for anything NER missed
    2. Normalize all field values
    3. Compute missing_fields
    """
    if text:
        raw = apply_rules(text, raw)

    result = {
        "destination": normalize_location(raw.get("destination")),
        "departure":   normalize_location(raw.get("departure")) or "current_location",
        "date":        normalize_date(raw.get("date")),
        "time":        normalize_time(raw.get("time")),
    }
    result["missing_fields"] = [
        f for f in ["destination", "date", "time"]
        if not result.get(f)
    ]
    return result


# ═══════════════════════════════════════════════════════════
# 4. INFERENCE
# ═══════════════════════════════════════════════════════════

def predict(nlp, text: str) -> dict:
    """Full pipeline: NER → rules → normalize → JSON."""
    doc = nlp(text)

    raw = {
        "destination": None,
        "departure":   None,
        "date":        None,
        "time":        None,
    }
    for ent in doc.ents:
        field = ent.label_.lower()
        if field in raw and raw[field] is None:
            raw[field] = ent.text

    return postprocess(raw, text)


# ═══════════════════════════════════════════════════════════
# 5. EVALUATION  (simple span-level accuracy)
# ═══════════════════════════════════════════════════════════

def evaluate(nlp, test_data: list) -> dict:
    """Quick F1-style eval on a held-out set."""
    tp = fp = fn = 0
    for text, ann in test_data:
        doc = nlp(text)
        pred_spans = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        true_spans = {(s, e, l) for s, e, l in ann["entities"]}
        tp += len(pred_spans & true_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


# ═══════════════════════════════════════════════════════════
# 6. MAIN
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
    import sys
    import os

    MODEL_PATH = "moviroo_model"

    # ── train or load ─────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        print(f"📦 Loading existing model from {MODEL_PATH}/")
        nlp = spacy.load(MODEL_PATH)
    else:
        # split 80/20 for quick eval
        random.shuffle(TRAIN_DATA)
        split     = int(len(TRAIN_DATA) * 0.8)
        train_set = TRAIN_DATA[:split]
        test_set  = TRAIN_DATA[split:]

        nlp = train_spacy_model(train_set, n_iter=120)

        if test_set:
            metrics = evaluate(nlp, test_set)
            print(f"📊 Eval → precision: {metrics['precision']}  "
                  f"recall: {metrics['recall']}  f1: {metrics['f1']}\n")

    # ── inference demo ────────────────────────────────────
    SEP = "─" * 58
    print(f"\n{SEP}")
    print("🧪  MOVIROO — Inference demo")
    print(SEP)

    for sentence in TEST_SENTENCES:
        result = predict(nlp, sentence)
        print(f"\n📝  {sentence}")
        print(f"     destination : {result['destination']}")
        print(f"     departure   : {result['departure']}")
        print(f"     date        : {result['date']}")
        print(f"     time        : {result['time']}")
        if result["missing_fields"]:
            print(f"     ⚠  missing  : {result['missing_fields']}")

    # ── interactive mode ──────────────────────────────────
    print(f"\n{SEP}")
    print("💬  Interactive mode — type a sentence (q to quit)")
    print(SEP)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("q", "quit", "exit"):
            break
        if not user_input:
            continue

        result = predict(nlp, user_input)
        print(json.dumps(result, ensure_ascii=False, indent=2))
