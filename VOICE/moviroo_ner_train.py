"""
╔══════════════════════════════════════════════════════════╗
║         MOVIROO — NER Training Pipeline                  ║
║   Fields: DESTINATION · DEPARTURE · DATE · TIME          ║
║   Languages: TN + FR + EN + AR                           ║
╚══════════════════════════════════════════════════════════╝

LEVEL 1 ─ spaCy NER  (fast, easy, good for prototype)
LEVEL 2 ─ BERT NER   (multilingual, production-ready)

Install:
    pip install spacy accelerate seqeval torch
    pip install "transformers>=4.41,<5.0"
    python -m spacy download xx_ent_wiki_sm
"""

# ─────────────────────────────────────────────────────────
# 0. SHARED TRAINING DATA  (TN / FR / EN / AR mixed)
# ─────────────────────────────────────────────────────────

TRAIN_DATA = [
    # ── Tunisian dialect ─────────────────────────────────
    ("nheb nemchi hammamet ghodwa 18h", {
        "entities": [
            (13, 22, "DESTINATION"),
            (23, 29, "DATE"),
            (30, 33, "TIME"),
        ]
    }),
    ("khodni lel matar tawwa min dar", {
        "entities": [
            (11, 16, "DESTINATION"),
            (17, 22, "DATE"),
            (27, 30, "DEPARTURE"),
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
            (39, 46, "TIME"),
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

    # ── French ───────────────────────────────────────────
    ("je veux aller a tunis demain matin", {
        "entities": [
            (16, 21, "DESTINATION"),
            (22, 28, "DATE"),
            (29, 34, "TIME"),
        ]
    }),
    ("taxi de la marsa vers le centre demain a 14h00", {
        "entities": [
            (8,  16, "DEPARTURE"),
            (22, 28, "DESTINATION"),
            (29, 35, "DATE"),
            (38, 43, "TIME"),
        ]
    }),
    ("aller de bizerte a nabeul mercredi a 9h", {
        "entities": [
            (9,  16, "DEPARTURE"),
            (19, 25, "DESTINATION"),
            (26, 34, "DATE"),
            (37, 39, "TIME"),
        ]
    }),
    ("taxi depuis mon bureau jusqu a sfax vendredi soir", {
        "entities": [
            (16, 22, "DEPARTURE"),
            (31, 35, "DESTINATION"),
            (36, 44, "DATE"),
            (45, 49, "TIME"),
        ]
    }),
    ("aller a sousse ce soir a 20h", {
        "entities": [
            (8,  14, "DESTINATION"),
            (15, 22, "DATE"),
            (25, 28, "TIME"),
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
    ("drop me at the hotel from my office friday morning", {
        "entities": [
            (15, 20, "DESTINATION"),
            (29, 35, "DEPARTURE"),
            (36, 42, "DATE"),
            (43, 50, "TIME"),
        ]
    }),
    ("book a ride to monastir next monday at 7h", {
        "entities": [
            (14, 22, "DESTINATION"),
            (23, 34, "DATE"),
            (38, 40, "TIME"),
        ]
    }),

    # ── Arabic ────────────────────────────────────────────
    ("من هنا إلى سوسة غدا في المساء", {
        "entities": [
            (3,  6,  "DEPARTURE"),
            (11, 15, "DESTINATION"),
            (16, 19, "DATE"),
            (23, 29, "TIME"),
        ]
    }),
    ("أريد سيارة إلى تونس الآن", {
        "entities": [
            (14, 18, "DESTINATION"),
            (19, 23, "DATE"),
        ]
    }),
    ("من المطار إلى صفاقس يوم الجمعة صباحا", {
        "entities": [
            (3,  9,  "DEPARTURE"),
            (13, 19, "DESTINATION"),
            (24, 31, "DATE"),
            (32, 37, "TIME"),
        ]
    }),
]


# ─────────────────────────────────────────────────────────
# 1. LEVEL 1 — spaCy NER
# ─────────────────────────────────────────────────────────

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random


def train_spacy(train_data, n_iter=80, output_path="moviroo_spacy_model"):
    nlp = spacy.blank("xx")
    ner = nlp.add_pipe("ner", last=True)

    for _, annotations in train_data:
        for _, _, label in annotations["entities"]:
            ner.add_label(label)

    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        try:
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        except Exception as e:
            print(f"  ⚠  Skipping: {text[:40]} → {e}")

    optimizer = nlp.initialize(lambda: examples)
    print("🚀 Training spaCy NER...\n")
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]

    with nlp.disable_pipes(*other_pipes):
        for iteration in range(n_iter):
            random.shuffle(examples)
            losses = {}
            batches = minibatch(examples, size=compounding(2.0, 8.0, 1.001))
            for batch in batches:
                nlp.update(batch, sgd=optimizer, drop=0.35, losses=losses)
            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d} | Loss NER: {losses['ner']:.4f}")

    nlp.to_disk(output_path)
    print(f"\n✅ spaCy model saved → {output_path}/")
    return nlp


def predict_spacy(nlp, text: str) -> dict:
    doc = nlp(text)
    result = {"destination": None, "departure": None, "date": None, "time": None, "raw_entities": []}
    for ent in doc.ents:
        result["raw_entities"].append({"text": ent.text, "label": ent.label_})
        field = ent.label_.lower()
        if field in result and result[field] is None:
            result[field] = ent.text
    return result


# ─────────────────────────────────────────────────────────
# 2. POST-PROCESSING
# ─────────────────────────────────────────────────────────

from datetime import date, timedelta
import re

DATE_MAP = {
    "lyoum": "today",      "lioum": "today",
    "tawwa": "now",
    "ghodwa": "tomorrow",  "ghodoa": "tomorrow",
    "ba3d ghodwa": "day_after_tomorrow",
    "el-jemaa": "friday",  "el-khamis": "thursday",
    "el-ahad": "sunday",   "el-itnin": "monday",
    "el-talata": "tuesday","el-arba3": "wednesday",
    "aujourd'hui": "today","demain": "tomorrow",
    "apres-demain": "day_after_tomorrow",
    "maintenant": "now",   "ce soir": "tonight",
    "lundi": "monday",     "mardi": "tuesday",
    "mercredi": "wednesday","jeudi": "thursday",
    "vendredi": "friday",  "samedi": "saturday",
    "dimanche": "sunday",
    "today": "today",      "tomorrow": "tomorrow",
    "tonight": "tonight",  "now": "now",
    "monday": "monday",    "tuesday": "tuesday",
    "wednesday": "wednesday","thursday": "thursday",
    "friday": "friday",    "saturday": "saturday",
    "sunday": "sunday",
    "الآن": "now",         "غدا": "tomorrow",
    "الجمعة": "friday",
}

TIME_MAP = {
    "sbeh": "morning",     "fel sbeh": "morning",
    "3achiya": "evening",  "fel 3achiya": "evening",
    "dhuhr": "noon",       "leyl": "night",
    "ba3d dhuhr": "afternoon",
    "matin": "morning",    "soir": "evening",
    "midi": "noon",        "nuit": "night",
    "apres-midi": "afternoon",
    "morning": "morning",  "evening": "evening",
    "noon": "noon",        "night": "night",
    "afternoon": "afternoon",
    "الصباح": "morning",   "المساء": "evening",
    "صباحا": "morning",
}

LOCATION_ALIASES = {
    "7amaamt": "Hammamet", "hammamet": "Hammamet",
    "sfax": "Sfax",        "sousse": "Sousse",
    "tunis": "Tunis",      "tnis": "Tunis",
    "monastir": "Monastir","nabeul": "Nabeul",
    "bizerte": "Bizerte",
    "matar": "Airport",    "aeroport": "Airport",
    "airport": "Airport",  "المطار": "Airport",
    "gare": "Train Station","mahatta": "Train Station",
    "beit": "Home",        "dar": "Home",
    "bureau": "Office",    "maktab": "Office",
    "el-maktab": "Office", "funduk": "Hotel",
    "hotel": "Hotel",      "centre": "City Center",
    "center": "City Center",
    "la marsa": "La Marsa","marsa": "La Marsa",
    "تونس": "Tunis",       "سوسة": "Sousse",
    "صفاقس": "Sfax",
}

DAY_NAMES = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}


def normalize_date(raw: str) -> str:
    if raw is None:
        return None
    key = raw.lower().strip()
    mapped = DATE_MAP.get(key, key)
    today = date.today()
    if mapped == "today" or mapped == "now":
        return today.isoformat()
    if mapped == "tomorrow":
        return (today + timedelta(days=1)).isoformat()
    if mapped == "day_after_tomorrow":
        return (today + timedelta(days=2)).isoformat()
    if mapped == "tonight":
        return today.isoformat()
    if mapped in DAY_NAMES:
        ahead = (DAY_NAMES[mapped] - today.weekday()) % 7 or 7
        return (today + timedelta(days=ahead)).isoformat()
    return mapped


def normalize_time(raw: str) -> str:
    if raw is None:
        return None
    key = raw.lower().strip()
    if key in TIME_MAP:
        return TIME_MAP[key]
    m = re.match(r"(\d{1,2})h(\d{0,2})", key)
    if m:
        h, mn = m.group(1), m.group(2) or "00"
        return f"{int(h):02d}:{mn.zfill(2)}"
    m = re.match(r"(\d{1,2}):(\d{2})", key)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    m = re.match(r"(\d{1,2})(am|pm)", key)
    if m:
        h = int(m.group(1))
        if m.group(2) == "pm" and h != 12:
            h += 12
        return f"{h:02d}:00"
    return raw


def normalize_location(raw: str) -> str:
    if raw is None:
        return None
    return LOCATION_ALIASES.get(raw.lower().strip(), raw.title())


def postprocess(raw_result: dict) -> dict:
    return {
        "destination":    normalize_location(raw_result.get("destination")),
        "departure":      normalize_location(raw_result.get("departure")) or "current_location",
        "date":           normalize_date(raw_result.get("date")),
        "time":           normalize_time(raw_result.get("time")),
        "missing_fields": [
            f for f in ["destination", "date", "time"]
            if not raw_result.get(f)
        ]
    }


# ─────────────────────────────────────────────────────────
# 3. LEVEL 2 — BERT NER
# ─────────────────────────────────────────────────────────

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset
import numpy as np

try:
    from seqeval.metrics import classification_report
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False
    print("⚠  seqeval not found — metrics will be skipped. pip install seqeval")

LABELS = [
    "O",
    "B-DESTINATION", "I-DESTINATION",
    "B-DEPARTURE",   "I-DEPARTURE",
    "B-DATE",        "I-DATE",
    "B-TIME",        "I-TIME",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL  = {i: l for l, i in LABEL2ID.items()}


def spacy_to_bio(text: str, entities: list, tokenizer) -> dict:
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=128,
    )
    offsets = encoding["offset_mapping"]

    char_labels = ["O"] * len(text)
    for start, end, label in entities:
        for i in range(start, min(end, len(text))):
            char_labels[i] = f"B-{label}" if i == start else f"I-{label}"

    labels = []
    prev_label = "O"
    for token_start, token_end in offsets:
        if token_start == token_end:
            labels.append(-100)
            continue
        tok_label = char_labels[token_start]
        if tok_label.startswith("I-") and not prev_label.endswith(tok_label[2:]):
            tok_label = "B-" + tok_label[2:]
        labels.append(LABEL2ID.get(tok_label, 0))
        prev_label = tok_label

    encoding["labels"] = labels
    encoding.pop("offset_mapping")
    return encoding


def build_bert_dataset(train_data, tokenizer):
    records = []
    for text, annotations in train_data:
        entities = annotations["entities"]
        enc = spacy_to_bio(text, entities, tokenizer)
        records.append(enc)
    return Dataset.from_list(records)


def compute_metrics(p):
    if not HAS_SEQEVAL:
        return {}
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(preds, labels):
        tl, tp = [], []
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                tl.append(ID2LABEL[label])
                tp.append(ID2LABEL[pred])
        true_labels.append(tl)
        true_preds.append(tp)
    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": report["weighted avg"]["precision"],
        "recall":    report["weighted avg"]["recall"],
        "f1":        report["weighted avg"]["f1-score"],
    }


def train_bert(train_data, output_dir="moviroo_bert_model", epochs=10, lr=2e-5):
    MODEL_NAME = "bert-base-multilingual-cased"
    print(f"📥 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("🔄 Building dataset...")
    dataset = build_bert_dataset(train_data, tokenizer)
    split    = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]

    print("🏗  Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        eval_strategy="epoch",        # ← fixed (was evaluation_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1" if HAS_SEQEVAL else "loss",
        logging_steps=5,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,   # ← fixed (tokenizer= is deprecated)
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\n🚀 Training BERT for {epochs} epochs...\n")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ BERT model saved → {output_dir}/")
    return trainer


def predict_bert(text: str, model_path="moviroo_bert_model") -> dict:
    from transformers import pipeline
    pipe = pipeline(
        "token-classification",
        model=model_path,
        aggregation_strategy="simple",
    )
    entities = pipe(text)
    result = {"destination": None, "departure": None, "date": None, "time": None}
    for ent in entities:
        field = ent["entity_group"].lower()
        if field in result and result[field] is None:
            result[field] = ent["word"].strip()
    return postprocess(result)


# ─────────────────────────────────────────────────────────
# 4. INTERACTIVE TEST LOOP
# ─────────────────────────────────────────────────────────

def interactive_loop(predict_fn):
    """
    Let the user type any sentence in the terminal and see
    extracted fields in real time.  Type 'q' to quit.
    """
    import json
    SEP = "─" * 58
    print(f"\n{SEP}")
    print("💬  Interactive test — type any sentence (q to quit)")
    print("    Examples:")
    print("      nheb nemchi hammamet ghodwa 18h")
    print("      je veux aller a tunis demain matin")
    print("      take me to the airport tonight at 22h")
    print("      من هنا إلى سوسة غدا في المساء")
    print(f"{SEP}\n")

    while True:
        try:
            user_input = input("📝 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break
        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            print("👋 Bye!")
            break

        result = predict_fn(user_input)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()


# ─────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Moviroo NER Trainer")
    parser.add_argument(
        "--level", choices=["spacy", "bert"], default="spacy",
        help="spacy = fast prototype | bert = multilingual production",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Launch interactive input loop after training",
    )
    cli = parser.parse_args()

    TEST_SENTENCES = [
        "nheb nemchi hammamet ghodwa 18h",
        "je veux aller a tunis demain matin",
        "take me to the airport tonight at 22h",
        "من هنا إلى سوسة غدا في المساء",
        "nemchi sfax min el-maktab el-khamis ba3d dhuhr",
    ]

    SEP = "─" * 58

    # ── spaCy ──────────────────────────────────────────────
    if cli.level == "spacy":
        nlp = train_spacy(TRAIN_DATA, n_iter=100)

        print(f"\n{SEP}")
        print("🧪 INFERENCE RESULTS (spaCy NER + post-processing)")
        print(SEP)
        for sentence in TEST_SENTENCES:
            raw   = predict_spacy(nlp, sentence)
            final = postprocess(raw)
            print(f"\n📝  {sentence}")
            print(f"     destination : {final['destination']}")
            print(f"     departure   : {final['departure']}")
            print(f"     date        : {final['date']}")
            print(f"     time        : {final['time']}")
            if final["missing_fields"]:
                print(f"     ⚠  missing  : {final['missing_fields']}")

        # always launch interactive loop for spaCy (fast to start)
        interactive_loop(lambda text: postprocess(predict_spacy(nlp, text)))

    # ── BERT ───────────────────────────────────────────────
    elif cli.level == "bert":
        import os
        MODEL_PATH = "moviroo_bert_model"

        if os.path.exists(MODEL_PATH):
            print(f"📦 Loading existing BERT model from {MODEL_PATH}/")
        else:
            train_bert(TRAIN_DATA, epochs=10)

        print(f"\n{SEP}")
        print("🧪 INFERENCE RESULTS (BERT NER + post-processing)")
        print(SEP)
        for sentence in TEST_SENTENCES:
            final = predict_bert(sentence, MODEL_PATH)
            print(f"\n📝  {sentence}")
            print(f"     destination : {final['destination']}")
            print(f"     departure   : {final['departure']}")
            print(f"     date        : {final['date']}")
            print(f"     time        : {final['time']}")
            if final["missing_fields"]:
                print(f"     ⚠  missing  : {final['missing_fields']}")

        if cli.interactive:
            interactive_loop(lambda text: predict_bert(text, MODEL_PATH))