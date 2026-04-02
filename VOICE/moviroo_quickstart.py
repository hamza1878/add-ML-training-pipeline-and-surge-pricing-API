"""
╔══════════════════════════════════════════════════════╗
║           MOVIROO — Quick Start Guide                ║
╚══════════════════════════════════════════════════════╝
"""

# ─── 1. INSTALL ───────────────────────────────────────────
"""
pip install spacy transformers datasets seqeval torch
python -m spacy download xx_ent_wiki_sm
"""

# ─── 2. TRAIN spaCy (Level 1 — prototype) ────────────────
"""
python moviroo_ner_train.py --level spacy
→ Saves model to: moviroo_spacy_model/
→ Expected F1: ~0.75 with 20 examples
→ Expected F1: ~0.90 with 200+ examples
"""

# ─── 3. TRAIN BERT (Level 2 — production 🔥) ─────────────
"""
python moviroo_ner_train.py --level bert
→ Saves model to: moviroo_bert_model/
→ Expected F1: ~0.92+ with 200 examples
→ Handles: TN + FR + EN + AR  automatically
"""

# ─── 4. USE IN YOUR APP ───────────────────────────────────
from moviroo_ner_train import (
    predict_spacy,
    predict_bert,
    postprocess,
)
import spacy

# ── Option A: spaCy (fast, ~5ms per request)
nlp   = spacy.load("moviroo_spacy_model")

def extract_spacy(user_input: str) -> dict:
    raw   = predict_spacy(nlp, user_input)
    return postprocess(raw)

# ── Option B: BERT (accurate, ~50ms per request)
def extract_bert(user_input: str) -> dict:
    return predict_bert(user_input, model_path="moviroo_bert_model")

# ─── 5. EXAMPLE OUTPUT ────────────────────────────────────
"""
Input:  "nheb nemchi hammamet ghodwa 18h"

Output: {
    "destination":    "Hammamet",
    "departure":      "current_location",
    "date":           "2025-04-16",
    "time":           "18:00",
    "missing_fields": []
}
─────────────────────────────────────────────────
Input:  "je veux aller à tunis"

Output: {
    "destination":    "Tunis",
    "departure":      "current_location",
    "date":           null,
    "time":           null,
    "missing_fields": ["date", "time"]
}
"""

# ─── 6. HOW TO GROW YOUR TRAINING DATA ───────────────────
"""
Minimum needed:
  ✦ spaCy:  50  examples → decent  |  200 → good  |  500 → great
  ✦ BERT:   100 examples → decent  |  300 → good  |  1000 → great

Tips to get more data fast:
  1. Collect real user inputs from your app (after launch)
  2. Use Claude/GPT to generate variations:
       "Generate 50 ride booking sentences in Tunisian dialect"
  3. Label with: https://doccano.github.io/doccano/
  4. Add augmentation: swap location names, change times

Format to add new examples:
  (
    "nheb nemchi nabeul ghodwa sbeh",
    {
        "entities": [
            (13, 19, "DESTINATION"),  # nabeul
            (20, 26, "DATE"),         # ghodwa
            (27, 31, "TIME"),         # sbeh
        ]
    }
  ),
"""

# ─── 7. PIPELINE ARCHITECTURE ─────────────────────────────
"""
🎤 Voice Input (optional)
      ↓  Whisper / Google STT
📝 Text
      ↓  NER Model (spaCy → BERT)
📦 Raw entities {destination, departure, date, time}
      ↓  Post-processing (normalizer.py)
✅ Structured JSON
      ↓  Missing fields?
💬 Dialog Manager → ask follow-up question
      ↓  All fields collected
🗺  Maps API (Google / OpenStreetMap) → coordinates
      ↓
💰 Pricing module (distance × rate)
      ↓
🚗 Booking confirmed
"""
