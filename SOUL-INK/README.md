# SOUL-INK — The AI That Listens

<p align="center"><img src="assets/SOUL-INK.png" alt="SOUL-INK logo" width="160"></p>

> *A 100% offline, multilingual AI mental health companion — no API key, no data stored, no account needed.*
> 
> Supports **English · French · Moroccan Darija (Arabic script + Franco-Darija) · Modern Standard Arabic**

---

## Overview

SOUL-INK is a local AI chatbot for emotional support. It listens, detects your emotional state in real time, and responds with empathy — proactively asking follow-up questions rather than waiting for you to pour everything out at once. It knows your local crisis line for 70+ countries and deploys it the moment it detects distress language.

⚠️ SOUL-INK is **not** a licensed therapist. It is a supportive tool — a bridge to help you in a moment of distress until a real professional can help.

---

## Features

| Feature | Detail |
|---|---|
| 💬 Multi-turn chat | Full conversation history within session |
| 🧠 Emotion detection | GoEmotions DistilBERT — 28 labels → 6 coarse groups |
| 🌍 4-language support | English, French, Darija (Franco + Arabic script), MSA Arabic |
| 🔄 Auto language detection | Detects script/vocabulary per message — switches fluid |
| 🆘 Crisis detection | Multilingual phrase matching + DODa dictionary enrichment |
| 📞 70+ country helplines | Personalised crisis card with local emergency + crisis line |
| 🌬 `/breathe` | 4-4-4-4 box breathing animated exercise |
| 🌿 `/ground` | 5-4-3-2-1 sensory grounding exercise |
| 📊 Mood timeline | Plotly chart of emotion shifts across session |
| 🔒 Zero persistence | Session-only — nothing stored when you leave |
| 🖥️ 100% local | Runs entirely on your machine — no internet after setup |

---

## Project Structure

```
SOUL-INK/
├── app.py                      # Main Streamlit application (UI + logic)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT
├── .gitignore
├── .streamlit/
│   └── config.toml             # Dark purple theme configuration
├── assets/
│   └── SOUL-INK.png            # Logo
└── notebooks/
    └── notebook_SOUL_INK.ipynb # Early Colab prototype (reference only)
```

No API keys or secrets are needed: everything runs locally.

---

## Architecture — Full Technical Breakdown

SOUL-INK is built as a **5-layer pipeline**. Each user message passes through all layers in sequence before a response is generated.

```
┌─────────────────────────────────────────────────────────┐
│                    User Message                         │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │  Layer 1: Safety      │  Crisis detection → immediate crisis card
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Layer 2: Language    │  Auto-detect → switch system prompt
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Layer 3: Emotion NLP │  GoEmotions DistilBERT → 28 emotions
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Layer 4: Theme NLP   │  spaCy NER + noun chunks → keywords
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Layer 5: LLM         │  Ollama local model → response
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  Streamlit Frontend   │  Chat UI + sidebar + exercises
        └───────────────────────┘
```

---

## Layer 1 — Safety & Crisis Detection

### What it does
Before any AI response is generated, every message is scanned for **crisis language** (self-harm, suicidal ideation, expressions of hopelessness). This is a hard gate — if triggered, the user sees their country's crisis helpline card *immediately*, before the LLM even runs.

### How it works
- **Keyword matching** across English, Darija, and Arabic: phrases like "I want to die", "bghit nmout", "أريد أن أموت" etc.
- **DODa dictionary enrichment** (`pydoda` library): scans the Moroccan semantic health corpus for Darija mental health vocabulary, expanding detection beyond a hardcoded list.
- **Country-aware response**: the crisis card displays the correct local emergency number and crisis line based on the user's selected country (from a 70+ country `HELPLINES_DB`).

### Crisis card system
```python
HELPLINES_DB = {
    "Morocco":        {"emergency": "15/19/177", "crisis": "0800 008 008", ...},
    "France":         {"emergency": "15/17/18",  "crisis": "3114", ...},
    "United States":  {"emergency": "911",        "crisis": "988", ...},
    # ... 70+ countries
}
```
The card renders in the **user's detected language** (English, French, Darija, or Arabic) with the correct flag, number, and local crisis line name.

---

## Layer 2 — Multilingual Detection & Routing

### Languages supported
| Code | Language | How detected |
|------|----------|-------------|
| `ar` | Modern Standard Arabic | Arabic Unicode block `\u0600–\u06FF`, pure script |
| `dar` | Moroccan Darija | Arabic Unicode OR Franco-Darija heuristics |
| `fr` | French | Accented characters (`éèêàç…`) + French vocabulary |
| `en` | English | Default fallback |

### Detection pipeline (priority order)
```
1. Arabic Unicode chars present?
     → YES: all chars non-ASCII? → "ar" (MSA)
     → YES: mixed with ASCII? → "dar" (Franco-Darija)
2. Franco-Darija patterns?
     → digit markers (3=ع, 7=ح, 9=ق) + Darija lexicon match → "dar"
3. French signals?
     → french-specific accented chars OR 2+ French function words → "fr"  
4. Default → "en"
```

### Franco-Darija (Arabizi) detection detail
Franco-Darija is Moroccan Arabic written in Latin characters using digit substitutes for sounds that don't exist in the Latin alphabet:
- `3` → ع (ain)
- `7` → ح (ha)
- `9` → ق (qaf)
- `kh` → خ, `gh` → غ, `ch` → ش

The detector uses a **two-signal heuristic**:
1. **Digit markers**: regex `\b\w*[379]\w*\b` (e.g. "3ndek", "7wija", "9al")
2. **Darija lexicon**: 50+ common Darija words ("wah", "bzzaf", "kayn", "daba", "mashi"…)

A match on either signal triggers Darija mode.

### Routing effect
Once a language is detected, the app switches **three things simultaneously**:
1. The **system prompt** sent to the LLM (personality, tone, and instructions in that language)
2. The **UI strings** (sidebar labels, placeholders, disclaimers, exercise text)
3. The **crisis card** language (the same emergency numbers, rendered in the user's language)

Users can also manually override with `/lang en`, `/lang fr`, `/lang dar`, or `/lang ar`.

---

## Layer 3 — Emotion Detection (NLP Model)

### Model: GoEmotions DistilBERT
- **Base model**: `distilbert-base-uncased` fine-tuned on Google's GoEmotions dataset
- **HuggingFace ID**: `joeddav/distilbert-base-uncased-go-emotions`
- **Task**: multi-label text classification → 28 fine-grained emotion labels
- **Runtime**: local inference via HuggingFace `transformers` + `PyTorch` — no API call

### 28 → 6 label coarsening
The 28 GoEmotions labels are grouped into 6 coarse emotional clusters for UX readability:

```python
EMO_GROUPS = {
    "joy":     {joy, amusement, excitement, gratitude, love, optimism, pride, relief...},
    "sadness": {sadness, disappointment, embarrassment, grief, remorse},
    "anger":   {anger, annoyance, disgust, disapproval},
    "fear":    {fear, nervousness, apprehension},
    "shame":   {guilt, shame},
    "neutral": {neutral, curiosity, realization, surprise, confusion},
}
```

### How emotion feeds the LLM
The top detected emotion is formatted into the LLM system prompt as `{emotion_context}`:
```
Current emotional state: sadness (primary), grief (detected)
Adjust your response to gently acknowledge and validate this emotional state first.
```
This ensures the LLM **always validates before advising** — a core principle of DBT and ACT.

---

## Layer 4 — Theme Extraction (spaCy NLP)

### Model: `en_core_web_sm`
- **Library**: spaCy 3.x
- **Pipeline components used**: tokenizer, POS tagger, NER (Named Entity Recognition), dependency parser

### What it extracts
From each user message, spaCy extracts **noun chunks** and **named entities** tagged as themes (e.g. "my exams", "my mum", "work stress"). These are:
1. Displayed in the sidebar **Themes** tracker for the user to see what's been discussed
2. Passed to the LLM context window as `{emotion_context}` to help the model give **contextually specific** responses

### DODa Darija Dictionary (`pydoda`)
For Darija messages, the app also queries the **DODa (Darija Open Dataset)** via the `pydoda` library. DODa provides a semantic lexicon of Moroccan Arabic health-related vocabulary, enriching crisis detection and theme extraction beyond what spaCy's English model handles.

---

## Layer 5 — Large Language Model (LLM)

### Runtime: Ollama (local model server)
Ollama runs entirely on the user's machine. It exposes a REST API at `http://localhost:11434` and serves any model the user has pulled. SOUL-INK sends requests to this local endpoint — **no internet required at inference time**.

```
SOUL-INK app ──► POST localhost:11434/api/chat ──► Ollama ──► model ──► response
```

### Supported models (user's choice)
| Model | Parameters | Description |
|-------|-----------|-------------|
| `gemma3:4b` | 4B | Google Gemma 3 — strong instruction following, multilingual |
| `gemma3:12b` | 12B | Larger Gemma — better nuance |
| `qwen3:4b` | 4B | Alibaba Qwen 3 — very strong on Arabic + French |
| `qwen3:8b` | 8B | Larger Qwen — best multilingual performance |
| `llama3` | 8B | Meta Llama 3 — general purpose English |
| `mistral` | 7B | Mistral — fast, good instruction following |

> **Recommended for multilingual (Arabic/French/Darija):** `qwen3:4b` or `qwen3:8b` — these are trained on significantly more multilingual data than Llama-family models.

### System Prompt Engineering
The LLM's behaviour is entirely controlled by the **system prompt** — a carefully engineered set of instructions that defines persona, conversational style, safety rules, and dynamic context.

**The system prompt has 4 dynamic sections:**

```
1. Persona definition      — "You are SOUL-INK, a warm caring AI friend..."
2. Conversational style    — "Be short, warm, proactive. 2-3 sentences max.
                              Acknowledge feeling first, then ask one specific
                              follow-up. When user says 'exams', pivot to a plan."
3. Safety rules            — "Never minimize feelings. Always validate first.
                              On crisis: give the crisis number immediately."
4. Dynamic context         — {emotion_context}: current detected emotion
                             {country_context}: user's local crisis number
```

The key design decision: **the LLM is instructed to behave like a caring friend, not a therapist**. Instead of long clinical paragraphs, it:
- Responds in 2–3 sentences
- Immediately acknowledges the feeling
- Always asks ONE specific follow-up question
- When given context ("exams", "my boss"), pivots to something actionable ("want to build a study plan together?")

Four separate system prompts exist — one per language (`SYSTEM_PROMPT`, `SYSTEM_PROMPT_FR`, `SYSTEM_PROMPT_DAR`, `SYSTEM_PROMPT_AR`) — each written natively in that language with culturally appropriate phrasing.

### Offline fallback
If Ollama is not running, the app **does not crash**. It falls back to a `dict`-based rule engine (`EMO_FALLBACK`) that returns a pre-written warm response keyed on the detected emotion. This works in all 4 languages.

---

## Frontend — Streamlit UI

### Stack
- **Streamlit 1.x** — Python-native web app framework
- **Custom CSS** — dark glassmorphism theme, RTL support for Arabic, animated breathing circle
- **Plotly** — mood timeline chart rendered with `px.bar`

### Key UI components
| Component | Description |
|-----------|-------------|
| Chat messages | `st.chat_message()` with avatar, timestamp, emotion tag |
| Language badge | Auto-detected language shown as a pill in sidebar |
| Helpline card | Rendered in sidebar when country is selected |
| Breathing exercise | CSS-animated pulsing circle with 4-4-4-4 cue text |
| Grounding exercise | Markdown card with 5-4-3-2-1 sensory steps |
| Mood chart | Plotly horizontal bar — emotion distribution over session |
| Mood timeline | Plotly line chart — emotion shift over message index |

### RTL Support
When Arabic (`ar`) is detected, a CSS class `.rtl-mode` is applied that sets `direction: rtl` on all chat messages, markdown blocks, and headings — making the layout flow properly for right-to-left Arabic text.

---

## Data Flow — End to End

```
User types: "je suis stressé à cause des examens"
    │
    ├─► Crisis check: no crisis phrases → pass
    │
    ├─► Language detect: French chars (é) + "je", "suis", "des" → lang = "fr"
    │         → Load SYSTEM_PROMPT_FR + French UI strings
    │
    ├─► GoEmotions DistilBERT:
    │         input: "je suis stressé à cause des examens"
    │         output: fear (0.61), nervousness (0.48) → coarse = "fear"
    │         → emotion_context = "Current state: fear/nervousness"
    │
    ├─► spaCy: noun chunks = ["examens"] → themes = ["examens"]
    │
    └─► Ollama (qwen3:4b) with system prompt:
              "Tu es SOUL-INK... [French persona]
               État émotionnel détecté: peur/stress
               Numéro de crise local: 3114 (France)"

              → Response: "Oh c'est vraiment stressant 💜 Les examens c'est une
                           pression énorme. T'as envie qu'on fasse un plan de
                           révisions ensemble, ou tu veux juste parler d'abord ?"
```

---

## Tech Stack Summary

```
Language        Python 3.10+
Frontend        Streamlit 1.x + Vanilla CSS (dark glassmorphism)
LLM Backend     Ollama (local) — Gemma3, Qwen3, Llama3, Mistral
Emotion NLP     HuggingFace Transformers — GoEmotions DistilBERT
                joeddav/distilbert-base-uncased-go-emotions
Deep Learning   PyTorch (CPU or GPU — auto-detected)
Theme NLP       spaCy en_core_web_sm (NER + noun chunks)
Darija Dict     pydoda / DODa open dataset (MIT license)
Charts          Plotly Express
Language detect Custom heuristics (Unicode + regex + lexicon)
```

---

## Dependencies

See [`requirements.txt`](./requirements.txt) for the full list with notes.

```txt
streamlit
plotly
requests
transformers   # optional — emotion detection
torch          # optional — emotion detection
spacy          # optional — theme extraction
pydoda         # optional — Darija crisis-phrase enrichment
```

> Note: SOUL-INK talks to Ollama over its local REST API (`http://localhost:11434`), so the `ollama` Python package itself is not required — only the Ollama application/server.

---

## Setup & Run

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Install & start Ollama
Download from [ollama.com](https://ollama.com/download), then:
```bash
ollama pull llama3         # the app starts on llama3
# optional, pick it in the sidebar afterwards:
# ollama pull qwen3:4b     # recommended for multilingual (Arabic/French/Darija)
# or: ollama pull gemma3:4b / mistral
ollama serve
```

### 3. Run SOUL-INK
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

> **Offline fallback:** If Ollama is not running, SOUL-INK still works with a warm, emotion-aware rule-based response in all 4 languages.
> **Optional NLP layers:** `transformers`/`torch` (emotion detection), `spacy` (theme extraction), and `pydoda` (Darija crisis enrichment) are all optional — the app degrades gracefully if any are missing.

---

## Privacy & Security

- **No database.** No accounts. No cookies. No cloud.
- **No external network calls while running** — the only HTTP request goes to `localhost:11434` (your local Ollama server). The emotion model and spaCy model are downloaded once during setup.
- All session data lives only in Streamlit's in-memory `session_state` — automatically wiped when the tab is closed or refreshed.
- The app never asks for your name, age, or any identifying information.

---

## Limitations

SOUL-INK is **not** a substitute for professional mental health care. It cannot diagnose, treat, or prescribe. The emotion detection model (GoEmotions) was trained on English social media text and may be less accurate on non-English or highly colloquial inputs. Crisis detection relies on keyword matching — it may miss subtle or indirect expressions of distress.

**If you are in crisis, please call your local emergency number.** SOUL-INK displays it automatically based on your country.

---

*Built for NavHacks 2025 · Extended with multilingual crisis support & conversational AI 2026*
