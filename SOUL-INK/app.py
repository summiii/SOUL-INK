"""
SOUL-INK — The AI That Listens
Multi-turn mental health chatbot powered by a LOCAL Ollama model.
100% offline · No API key · Session-only · Anonymous by design.
"""

import io, json, time, re, requests
from datetime import datetime
from collections import Counter

import streamlit as st
import plotly.graph_objects as go

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="SOUL-INK — The AI That Listens",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS (full dark) ──────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  :root {
    --ink:    #a855f7;
    --ink2:   #7c3aed;
    --ink3:   #c084fc;
    --glow:   rgba(168,85,247,0.35);
    --bg:     #08060d;
    --bg2:    #0f0b1a;
    --card:   rgba(168,85,247,0.07);
    --border: rgba(168,85,247,0.22);
    --text:   #e9d5ff;
    --sub:    #9d87b0;
    --green:  #86efac;
    --red:    #fca5a5;
  }

  html, body { background: var(--bg) !important; }

  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewBlockContainer"],
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"],
  [data-testid="block-container"],
  .main, .main .block-container {
    background: radial-gradient(ellipse 140% 80% at 15% 8%,
      #1b0f28 0%, #0d0918 45%, #08060d 100%) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important;
  }

  footer,
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] { visibility: hidden !important; height: 0 !important; }

  [data-testid="stSidebar"],
  [data-testid="stSidebar"] > div,
  [data-testid="stSidebarContent"] {
    background: #0b0718 !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  [data-testid="stBottom"],
  [data-testid="stBottom"] > div,
  .stChatFloatingInputContainer,
  .stChatInputContainer,
  section[data-testid="stBottom"] {
    background: #08060d !important;
    border-top: 1px solid var(--border) !important;
  }

  [data-testid="stChatInput"],
  [data-testid="stChatInput"] > div {
    background: #0f0b1a !important;
  }
  [data-testid="stChatInput"] textarea {
    background: rgba(168,85,247,0.09) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.97rem !important;
    caret-color: var(--ink3) !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: var(--ink) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
    outline: none !important;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: var(--sub) !important; }
  [data-testid="stChatInput"] button {
    background: var(--ink) !important;
    border-radius: 10px !important;
  }

  input, textarea, select,
  [data-baseweb="input"] input,
  [data-baseweb="textarea"] textarea,
  .stTextInput input,
  .stTextArea textarea {
    background: #1a1228 !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
  }
  input::placeholder, textarea::placeholder { color: var(--sub) !important; }

  .stAlert,
  [data-testid="stNotification"],
  div[role="alert"],
  .stInfo, .element-container .stInfo > div {
    background: rgba(168,85,247,0.10) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
  }
  [data-testid="stNotification"][kind="info"],
  div[data-baseweb="notification"] {
    background: rgba(124,58,237,0.15) !important;
    border-color: var(--ink2) !important;
  }

  .stSelectbox [data-baseweb="select"] > div {
    background: #1a1228 !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
  }

  h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #e9d5ff, #a855f7);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0 !important;
  }
  h2, h3, h4 { color: var(--ink3) !important; font-weight: 600 !important; }
  label, .stMarkdown p { color: var(--text) !important; }

  [data-testid="stChatMessage"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    margin-bottom: 0.6rem !important;
    padding: 0.8rem 1.1rem !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, var(--ink2), var(--ink)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px var(--glow) !important;
  }

  [data-testid="metric-container"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
  }
  [data-testid="stMetricValue"],
  [data-testid="stMetricLabel"] { color: var(--text) !important; }

  hr { border-color: var(--border) !important; }

  .streamlit-expanderHeader,
  [data-testid="stExpander"] summary {
    background: var(--card) !important;
    border-radius: 10px !important;
    color: var(--ink3) !important;
  }
  [data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--ink2); border-radius: 4px; }

  .crisis-banner {
    background: linear-gradient(135deg, rgba(220,38,38,0.18), rgba(168,85,247,0.15));
    border: 1.5px solid rgba(252,165,165,0.5);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text);
  }

  @keyframes breathe {
    0%   { transform: scale(1);    box-shadow: 0 0 0 0 var(--glow); }
    50%  { transform: scale(1.35); box-shadow: 0 0 0 40px rgba(168,85,247,0); }
    100% { transform: scale(1);    box-shadow: 0 0 0 0 rgba(168,85,247,0); }
  }
  .breathe-circle {
    width: 110px; height: 110px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--ink), var(--ink2));
    animation: breathe 4s ease-in-out infinite;
    margin: 1.5rem auto;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 0.85rem; font-weight: 600; text-align: center;
    line-height: 1.3;
  }

  .disclaimer {
    font-size: 0.78rem; color: var(--sub);
    background: rgba(168,85,247,0.06); border: 1px solid var(--border);
    border-radius: 999px; padding: 0.3rem 0.9rem;
    display: inline-block; margin-bottom: 0.8rem;
  }
  .tag {
    display: inline-block;
    background: rgba(168,85,247,0.18); border: 1px solid var(--border);
    border-radius: 999px; padding: 2px 10px;
    font-size: 0.78rem; margin: 2px; color: var(--ink3);
  }
  .status-online {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(134,239,172,0.12); border: 1px solid rgba(134,239,172,0.35);
    border-radius: 999px; padding: 4px 12px;
    font-size: 0.82rem; color: #86efac; font-weight: 500;
  }
  .status-offline {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(252,165,165,0.12); border: 1px solid rgba(252,165,165,0.35);
    border-radius: 999px; padding: 4px 12px;
    font-size: 0.82rem; color: #fca5a5; font-weight: 500;
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
  }
  .dot-pulse { animation: pulse-dot 1.4s ease-in-out infinite; }

  /* ── Hero header ── */
  .hero {
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 1.1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
  }
  .hero-badge {
    flex-shrink: 0;
    width: 52px; height: 52px;
    border-radius: 16px;
    background: linear-gradient(135deg, var(--ink2), var(--ink));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 4px 18px var(--glow);
  }
  .hero-text {
    font-size: 1.7rem; font-weight: 700;
    background: linear-gradient(135deg, #e9d5ff, #a855f7);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    line-height: 1.25;
  }
  .hero-sub {
    font-size: 0.92rem; font-weight: 400; margin-top: 0.15rem;
    color: var(--sub) !important;
    -webkit-text-fill-color: var(--sub) !important;
    background: none !important;
  }
  .hero-sub p { margin: 0 !important; }

  /* ── User vs assistant message bubbles ── */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="user" i]) {
    background: rgba(124,58,237,0.13) !important;
    border-color: rgba(168,85,247,0.30) !important;
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label*="assistant" i]) {
    background: rgba(168,85,247,0.05) !important;
    border-color: var(--border) !important;
  }

  /* ── Architecture / pipeline panel ── */
  .pipeline-step {
    display: flex; align-items: flex-start; gap: 0.7rem;
    padding: 0.55rem 0;
    border-bottom: 1px dashed var(--border);
  }
  .pipeline-step:last-child { border-bottom: none; }
  .pipeline-num {
    flex-shrink: 0;
    width: 26px; height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--ink2), var(--ink));
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; color: white;
  }
  .pipeline-title { font-weight: 600; color: var(--ink3); font-size: 0.88rem; }
  .pipeline-desc  { font-size: 0.8rem; color: var(--sub); line-height: 1.4; margin-top: 1px; }

  /* ── RTL support for Arabic ── */
  .rtl-mode .stChatMessage,
  .rtl-mode .stMarkdown,
  .rtl-mode p,
  .rtl-mode h1, .rtl-mode h2, .rtl-mode h3 {
    direction: rtl !important;
    text-align: right !important;
  }
  .rtl-mode { direction: rtl; }
  .rtl-mode .hero { flex-direction: row-reverse; }
  .rtl-mode .hero-text { text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
OLLAMA_BASE = "http://localhost:11434"

AVAILABLE_MODELS = [
    "gemma3:4b",
    "gemma3:1b",
    "gemma3:12b",
    "qwen3:4b",
    "qwen3:8b",
    "deepseek-r1:8b",
    "llama3.2",
    "llama3",
    "mistral",
    "phi3",
]

CRISIS_PHRASES_EN = {
    "suicide", "kill myself", "end it all", "end my life",
    "self harm", "self-harm", "cutting myself", "no reason to live",
    "overdose", "i want to die", "want to die", "want to end", "i want to kill",
    "i don't want to be here", "i dont want to be here", "hurt myself",
    "not worth living", "nothing to live for", "take my own life",
}

# ── DODa / PyDODa — open-source Darija dictionary ─────────────────────────────
# https://github.com/saad-out/pydoda  |  pip install pydoda
# Licence: MIT  |  Dataset: Darija Open Dataset (DODa), 21 000+ entries
@st.cache_resource(show_spinner=False)
def _load_doda_crisis_terms() -> set:
    """
    Build a set of Darija crisis phrases sourced from:
      1. Hardcoded anchors (Franco-Darija + Arabic script)
      2. DODa open dataset via pydoda: scans health/emotions categories
         and keeps any Darija entry whose English meaning overlaps
         with crisis-adjacent keywords.
    Falls back gracefully if pydoda is not installed.
    DODa: https://github.com/darija-open-dataset/dataset (MIT licence)
    """
    base_crisis = {
        # Franco-Darija hardcoded anchors
        "nbghi nmout", "nbghi nqtl rassi", "nqtl rassi", "ma3andish sb",
        "khsni nmout", "bghit nmout", "ma3ditsh nghder", "bghit nswi",
        "bghit nmut", "radi nmut", "3lach nbqa", "ma3aditsh nghder",
        "bghit nkhellas", "mabghitsh nbqa", "3ndash sabab",
        "radi nqtel rassi", "kanfker f mout", "bghit mout",
        "mabghitch nbqa", "3lach khasni nbqa", "radi ndhreb rassi",
        # Arabic-script Darija
        "\u0628\u063a\u064a\u062a \u0646\u0645\u0648\u062a",
        "\u0646\u0628\u063a\u064a \u0646\u0645\u0648\u062a",
        "\u0646\u0642\u062a\u0644 \u0631\u0627\u0633\u064a",
        "\u0645\u0627 \u0639\u0646\u062f\u064a\u0634 \u0633\u0628\u0628",
        "\u0628\u063a\u064a\u062a \u0646\u0642\u062a\u0644 \u0631\u0627\u0633\u064a",
        "\u0628\u063a\u064a\u062a \u0646\u0645\u0648\u062a",
        "\u0631\u0627\u062f\u064a \u0646\u0645\u0648\u062a",
    }
    _CRISIS_EN_KEYWORDS = {
        "death", "die", "dead", "dying", "suicide", "kill", "killing",
        "pain", "hurt", "hurting", "sad", "sadness", "hopeless", "desperate",
        "suffer", "suffering", "bleed", "cut", "alone", "loneliness",
        "empty", "worthless", "useless", "depressed", "depression",
        "anxiety", "fear", "grief", "mourn", "loss", "trauma",
    }
    try:
        from pydoda import Category
        for cat_type, cat_name in [("semantic", "emotions"), ("semantic", "health")]:
            try:
                cat = Category(cat_type, cat_name)
                # Approach 1: forward translate (English -> Darija)
                for kw in _CRISIS_EN_KEYWORDS:
                    try:
                        dar = cat.get_darija_translation(kw)
                        if dar and isinstance(dar, str) and len(dar.strip()) > 1:
                            base_crisis.add(dar.lower().strip())
                    except Exception:
                        pass
                # Approach 2: scan DODa Darija words (n1/n2/n3 columns) via __category attr
                try:
                    inner = cat._Category__category  # pandas DataFrame
                    # Identify Darija columns (n1, n2, n3...) and English column
                    en_col = next((c for c in inner.columns if c.lower() in ("en", "english", "eng")), None)
                    dar_cols = [c for c in inner.columns if c.lower().startswith("n") and c not in ("name",)]
                    if en_col:
                        for _, row in inner.iterrows():
                            en_val = str(row.get(en_col, "") or "").lower()
                            if any(kw in en_val for kw in _CRISIS_EN_KEYWORDS):
                                for dc in dar_cols:
                                    dv = str(row.get(dc, "") or "").strip()
                                    if len(dv) > 1 and dv.lower() != "nan":
                                        base_crisis.add(dv.lower())
                except Exception:
                    pass
            except Exception:
                pass
    except ImportError:
        pass  # pydoda not installed — base_crisis is still sufficient
    return base_crisis


# ── Worldwide crisis helpline database ────────────────────────────────────────
# Sources: OpenCounseling.com, Befrienders Worldwide, findahelpline.com
HELPLINES_DB = {
    "Algeria":          {"flag":"🇩🇿","emergency":"3042 / 43",              "crisis":"0021 3983 2000 58","name":"Centre Anti-Poison & Urgences",          "web":"https://findahelpline.com/dz"},
    "Argentina":        {"flag":"🇦🇷","emergency":"911",                    "crisis":"135",              "name":"Centro de Asistencia al Suicida",       "web":"https://www.asistenciaalsuicida.org.ar"},
    "Australia":        {"flag":"🇦🇺","emergency":"000",                    "crisis":"13 11 14",         "name":"Lifeline Australia",                    "web":"https://www.lifeline.org.au"},
    "Austria":          {"flag":"🇦🇹","emergency":"112",                    "crisis":"142",              "name":"Telefonseelsorge (24/7)",                "web":"https://www.telefonseelsorge.at"},
    "Belgium":          {"flag":"🇧🇪","emergency":"112",                    "crisis":"0800 32 123",      "name":"Centrum ter Preventie van Zelfdoding",   "web":"https://www.zelfmoord1813.be"},
    "Bolivia":          {"flag":"🇧🇴","emergency":"911",                    "crisis":"3911270",          "name":"CIAPS Bolivia",                         "web":"https://findahelpline.com/bo"},
    "Brazil":           {"flag":"🇧🇷","emergency":"190",                    "crisis":"188",              "name":"CVV — Centro de Valorizacao da Vida",    "web":"https://www.cvv.org.br"},
    "Canada":           {"flag":"🇨🇦","emergency":"911",                    "crisis":"988",              "name":"Canada Suicide Crisis Helpline",        "web":"https://988.ca"},
    "Chile":            {"flag":"🇨🇱","emergency":"911",                    "crisis":"600 360 7777",     "name":"ACHS Mental Health Line",               "web":"https://findahelpline.com/cl"},
    "China":            {"flag":"🇨🇳","emergency":"110",                    "crisis":"800-810-1117",     "name":"Beijing Suicide Research & Prevention", "web":"http://www.crisis.org.cn"},
    "Colombia":         {"flag":"🇨🇴","emergency":"123",                    "crisis":"106",              "name":"Linea 106 Ministerio de Salud",         "web":"https://findahelpline.com/co"},
    "Croatia":          {"flag":"🇭🇷","emergency":"112",                    "crisis":"116 123",          "name":"Samaritans Croatia",                    "web":"https://findahelpline.com/hr"},
    "Cyprus":           {"flag":"🇨🇾","emergency":"112",                    "crisis":"8000 7773",        "name":"Hope for Life (24/7)",                  "web":"https://findahelpline.com/cy"},
    "Czech Republic":   {"flag":"🇨🇿","emergency":"112",                    "crisis":"116 123",          "name":"Linka Bezpeci",                         "web":"https://www.linkabezpeci.cz"},
    "Denmark":          {"flag":"🇩🇰","emergency":"112",                    "crisis":"70 201 201",       "name":"Livslinien",                            "web":"https://www.livslinien.dk"},
    "Egypt":            {"flag":"🇪🇬","emergency":"122",                    "crisis":"08008880700",      "name":"Nada Foundation",                       "web":"https://findahelpline.com/eg"},
    "Estonia":          {"flag":"🇪🇪","emergency":"112",                    "crisis":"3726558088",       "name":"Estonian Crisis Line",                  "web":"https://findahelpline.com/ee"},
    "Finland":          {"flag":"🇫🇮","emergency":"112",                    "crisis":"09 2525 0111",     "name":"MIELI Mental Health Finland (24/7)",     "web":"https://mieli.fi"},
    "France":           {"flag":"🇫🇷","emergency":"15 / 112",               "crisis":"3114",             "name":"Numero National Prevention Suicide",    "web":"https://www.3114.fr"},
    "Germany":          {"flag":"🇩🇪","emergency":"112",                    "crisis":"0800 111 0 111",   "name":"Telefonseelsorge (24/7)",                "web":"https://www.telefonseelsorge.de"},
    "Ghana":            {"flag":"🇬🇭","emergency":"999",                    "crisis":"0800 111 222",     "name":"Mental Health Authority Ghana",         "web":"https://findahelpline.com/gh"},
    "Greece":           {"flag":"🇬🇷","emergency":"166",                    "crisis":"10306",            "name":"KLIMAKA Suicide Hotline",               "web":"https://www.klimaka.org.gr"},
    "Hungary":          {"flag":"🇭🇺","emergency":"112",                    "crisis":"116 123",          "name":"Befrienders Hungary",                   "web":"https://findahelpline.com/hu"},
    "India":            {"flag":"🇮🇳","emergency":"112",                    "crisis":"9152987821",       "name":"iCall / Vandrevala Foundation",         "web":"https://icallhelpline.org"},
    "Indonesia":        {"flag":"🇮🇩","emergency":"119",                    "crisis":"119 ext 8",        "name":"Into The Light Indonesia",              "web":"https://www.intothelightid.org"},
    "Iran":             {"flag":"🇮🇷","emergency":"115",                    "crisis":"1480",             "name":"Social Emergency Hotline",              "web":"https://findahelpline.com/ir"},
    "Ireland":          {"flag":"🇮🇪","emergency":"999",                    "crisis":"116 123",          "name":"Samaritans Ireland (24/7)",              "web":"https://www.samaritans.org"},
    "Israel":           {"flag":"🇮🇱","emergency":"101",                    "crisis":"1201",             "name":"ERAN Crisis Line (24/7)",                "web":"https://www.eran.org.il"},
    "Italy":            {"flag":"🇮🇹","emergency":"118",                    "crisis":"800 27 43 47",     "name":"Telefono Amico",                        "web":"https://www.telefonoamico.it"},
    "Japan":            {"flag":"🇯🇵","emergency":"110",                    "crisis":"0120-783-556",     "name":"Inochi no Denwa (24/7)",                "web":"https://findahelpline.com/jp"},
    "Jordan":           {"flag":"🇯🇴","emergency":"911",                    "crisis":"110",              "name":"Jordan Ministry of Health",             "web":"https://findahelpline.com/jo"},
    "Kenya":            {"flag":"🇰🇪","emergency":"999",                    "crisis":"0800 720 603",     "name":"Befrienders Kenya",                     "web":"https://www.befrienderskenya.org"},
    "Kuwait":           {"flag":"🇰🇼","emergency":"112",                    "crisis":"94 06 93 04",      "name":"Kuwait Crisis Hotline",                 "web":"https://findahelpline.com/kw"},
    "Lebanon":          {"flag":"🇱🇧","emergency":"140",                    "crisis":"1564",             "name":"Embrace Lebanon (24/7)",                "web":"https://www.embracelebanon.org"},
    "Lithuania":        {"flag":"🇱🇹","emergency":"112",                    "crisis":"8 800 28 888",     "name":"Jaunimo Linija",                        "web":"https://www.jaunimolinija.lt"},
    "Luxembourg":       {"flag":"🇱🇺","emergency":"112",                    "crisis":"352 45 45 45",     "name":"SOS Detresse",                          "web":"https://sosdetresse.lu"},
    "Malaysia":         {"flag":"🇲🇾","emergency":"999",                    "crisis":"15999",            "name":"Befrienders KL (24/7)",                 "web":"https://www.befrienders.org.my"},
    "Malta":            {"flag":"🇲🇹","emergency":"112",                    "crisis":"179",              "name":"Supportline Malta (24/7)",               "web":"https://www.supportline.org.mt"},
    "Mauritius":        {"flag":"🇲🇺","emergency":"112",                    "crisis":"800 93 93",        "name":"BAHAI Crisis Line",                     "web":"https://findahelpline.com/mu"},
    "Mexico":           {"flag":"🇲🇽","emergency":"911",                    "crisis":"800 290 0024",     "name":"SAPTEL (24/7)",                         "web":"https://www.saptel.org.mx"},
    "Morocco":          {"flag":"🇲🇦","emergency":"15 (SAMU) / 19 (Police)","crisis":"141 / ANARUZ: 0800 008 008","name":"Urgences medicales Maroc / ANARUZ","web":"https://findahelpline.com/ma"},
    "Netherlands":      {"flag":"🇳🇱","emergency":"112",                    "crisis":"0800 0113",        "name":"113 Zelfmoordpreventie (24/7)",          "web":"https://www.113.nl"},
    "New Zealand":      {"flag":"🇳🇿","emergency":"111",                    "crisis":"1737",             "name":"Need to Talk? (24/7 text/call)",         "web":"https://1737.org.nz"},
    "Nigeria":          {"flag":"🇳🇬","emergency":"199",                    "crisis":"0800 800 2000",    "name":"NEEM Foundation",                       "web":"https://neemfoundation.org"},
    "Norway":           {"flag":"🇳🇴","emergency":"113",                    "crisis":"116 123",          "name":"Kirkens SOS (24/7)",                    "web":"https://www.kirkens-sos.no"},
    "Pakistan":         {"flag":"🇵🇰","emergency":"1122",                   "crisis":"0311-7786264",     "name":"Umang Pakistan",                        "web":"https://umang.com.pk"},
    "Peru":             {"flag":"🇵🇪","emergency":"911",                    "crisis":"0800 4 1212",      "name":"Centro de Salud Mental Comunitaria",    "web":"https://findahelpline.com/pe"},
    "Philippines":      {"flag":"🇵🇭","emergency":"911",                    "crisis":"1553",             "name":"In Touch Crisis Line (24/7)",            "web":"https://www.intouch.org.ph"},
    "Poland":           {"flag":"🇵🇱","emergency":"112",                    "crisis":"116 123",          "name":"Telefon Zaufania (24/7)",                "web":"https://findahelpline.com/pl"},
    "Portugal":         {"flag":"🇵🇹","emergency":"112",                    "crisis":"213 544 545",      "name":"SOS Voz Amiga (24/7)",                  "web":"https://www.sosvozamiga.org"},
    "Qatar":            {"flag":"🇶🇦","emergency":"999",                    "crisis":"16000",            "name":"NHCC Mental Health",                    "web":"https://findahelpline.com/qa"},
    "Romania":          {"flag":"🇷🇴","emergency":"112",                    "crisis":"0800 801 200",     "name":"Asociatia Telefonul Copilului",          "web":"https://www.telefonulcopilului.ro"},
    "Russia":           {"flag":"🇷🇺","emergency":"112",                    "crisis":"8-800-2000-122",   "name":"Telefon Doveriya",                      "web":"https://findahelpline.com/ru"},
    "Saudi Arabia":     {"flag":"🇸🇦","emergency":"911",                    "crisis":"920033360",        "name":"Ministry of Health Hotline",            "web":"https://findahelpline.com/sa"},
    "Singapore":        {"flag":"🇸🇬","emergency":"995",                    "crisis":"1800 221 4444",    "name":"Samaritans of Singapore (24/7)",         "web":"https://www.sos.org.sg"},
    "South Africa":     {"flag":"🇿🇦","emergency":"10111",                  "crisis":"0800 567 567",     "name":"SADAG (24/7)",                          "web":"https://www.sadag.org"},
    "South Korea":      {"flag":"🇰🇷","emergency":"119",                    "crisis":"1393",             "name":"Korea Suicide Prevention Hotline",      "web":"https://www.spckorea.or.kr"},
    "Spain":            {"flag":"🇪🇸","emergency":"112",                    "crisis":"024",              "name":"Linea de Atencion a la Conducta Suicida","web":"https://findahelpline.com/es"},
    "Sweden":           {"flag":"🇸🇪","emergency":"112",                    "crisis":"90101",            "name":"Mind Sjalvmordslinjen (24/7)",           "web":"https://mind.se/hitta-hjalp/sjalvmordslinjen"},
    "Switzerland":      {"flag":"🇨🇭","emergency":"144",                    "crisis":"143",              "name":"Die Dargebotene Hand (24/7)",            "web":"https://www.143.ch"},
    "Taiwan":           {"flag":"🇹🇼","emergency":"119",                    "crisis":"1925",             "name":"Taiwan Lifeline Int'l (24/7)",          "web":"https://www.life1925.org.tw"},
    "Thailand":         {"flag":"🇹🇭","emergency":"1669",                   "crisis":"02 713 6793",      "name":"Samaritans of Thailand (24/7)",         "web":"https://findahelpline.com/th"},
    "Tunisia":          {"flag":"🇹🇳","emergency":"197",                    "crisis":"73 211 211",       "name":"Association AMEN",                      "web":"https://findahelpline.com/tn"},
    "Turkey":           {"flag":"🇹🇷","emergency":"112",                    "crisis":"182",              "name":"Mental Health Hotline (24/7)",           "web":"https://findahelpline.com/tr"},
    "Uganda":           {"flag":"🇺🇬","emergency":"999",                    "crisis":"0800 21 21 21",    "name":"Mental Health Uganda",                  "web":"https://findahelpline.com/ug"},
    "United Arab Emirates":{"flag":"🇦🇪","emergency":"999",                 "crisis":"800 46342",        "name":"NLIDES Counselling",                    "web":"https://findahelpline.com/ae"},
    "United Kingdom":   {"flag":"🇬🇧","emergency":"999",                    "crisis":"116 123",          "name":"Samaritans UK (24/7)",                  "web":"https://www.samaritans.org"},
    "United States":    {"flag":"🇺🇸","emergency":"911",                    "crisis":"988",              "name":"988 Suicide & Crisis Lifeline (24/7)",  "web":"https://988lifeline.org"},
    "Zambia":           {"flag":"🇿🇲","emergency":"999",                    "crisis":"116",              "name":"Befrienders Zambia",                    "web":"https://findahelpline.com/zm"},
    "Zimbabwe":         {"flag":"🇿🇼","emergency":"999",                    "crisis":"0804 800 5000",    "name":"Friendship Bench Zimbabwe",             "web":"https://www.friendshipbenchzimbabwe.org"},
    # sentinel
    "Not specified":    {"flag":"🌍","emergency":"112 / 911",               "crisis":"findahelpline.com","name":"Find A Helpline (worldwide)",           "web":"https://findahelpline.com"},
}

COUNTRY_LIST = sorted(k for k in HELPLINES_DB if k != "Not specified") + ["Not specified"]


def build_helpline_card(country: str, lang: str = "en") -> str:
    """Return a formatted crisis-resource markdown string for a given country."""
    data      = HELPLINES_DB.get(country, HELPLINES_DB["Not specified"])
    flag      = data["flag"]
    emergency = data["emergency"]
    crisis    = data["crisis"]
    name      = data["name"]
    web       = data["web"]

    if lang == "dar":
        return (
            f"🆘 **Ila kenti f kh tar f had l'lahda, 3ayyet daba: {emergency}**\n\n"
            f"📞 **Kh toot dyal d3em (libre, 24/7):**\n"
            f"- {flag} **{name}** — **{crisis}**\n"
            f"- 🌐 [{web}]({web})\n"
            f"- 🌍 [findahelpline.com](https://findahelpline.com) — directory 3alami\n\n"
            "Nta/Nti mawahed/a. Had l'khtwa li dartiha belli tkellmti hiya l'khtwa dyal sha9a. 💜"
        )
    elif lang == "ar":
        return (
            f"🆘 **إذا كنت في خطر فوري، اتصل بالطوارئ الآن: {emergency}**\n\n"
            f"📞 **خط الأزمة المجاني (24/7):**\n"
            f"- {flag} **{name}** — اتصل أو أرسل رسالة إلى **{crisis}**\n"
            f"- 🌐 [{web}]({web})\n"
            f"- 🌍 [findahelpline.com](https://findahelpline.com) — دليل عالمي\n\n"
            "لست وحدك/ي. طلب المساعدة شجاعة حقيقية. 💜"
        )
    elif lang == "fr":
        return (
            f"🆘 **Si tu es en danger immédiat, appelle les secours maintenant : {emergency}**\n\n"
            f"📞 **Ligne de crise gratuite (24h/24, 7j/7) :**\n"
            f"- {flag} **{name}** — appelle ou envoie un SMS au **{crisis}**\n"
            f"- 🌐 [{web}]({web})\n"
            f"- 🌍 [findahelpline.com](https://findahelpline.com) — annuaire mondial\n\n"
            "Tu n'es pas seul/e. Tendre la main, c'est du courage. 💜"
        )
    else:
        return (
            f"🆘 **If you're in immediate danger, call emergency services now: {emergency}**\n\n"
            f"📞 **Free crisis line (24/7):**\n"
            f"- {flag} **{name}** — call or text **{crisis}**\n"
            f"- 🌐 [{web}]({web})\n"
            f"- 🌍 [findahelpline.com](https://findahelpline.com) — worldwide directory\n\n"
            "You are not alone. Reaching out is an act of real courage. 💜"
        )


EMO_GROUPS = {
    "joy":     {"joy","amusement","excitement","gratitude","love","optimism","pride","relief","admiration","caring"},
    "sadness": {"sadness","disappointment","embarrassment","grief","remorse"},
    "anger":   {"anger","annoyance","disgust","disapproval"},
    "fear":    {"fear","nervousness","apprehension"},
    "shame":   {"guilt","shame"},
    "neutral": {"neutral","curiosity","realization","surprise","confusion"},
}

SYSTEM_PROMPT = """You are SOUL-INK — a warm, caring AI companion for mental health support. Think of yourself as a thoughtful friend who really listens.

⚠️ What you are NOT: You are NOT a licensed therapist or medical professional. You cannot diagnose or replace professional care. If the app stops working, always remind the user to call their local crisis line (shown at the start of the session).

Your conversational style — THIS IS CRITICAL:
- Be short, warm, and natural. 2–3 sentences max per response unless the user needs more.
- NEVER wait for the user to give you everything. Respond to what they’ve shared, then ask ONE specific follow-up to draw more out.
- Acknowledge their feeling first (“oh that sounds really hard”), THEN ask what’s going on (“what happened?”).
- When they give you context (e.g. “exams”, “my mum”, “work”), immediately pivot to something actionable and specific: “that’s rough — do you want to talk through a plan for that?” or “what’s the hardest part of it right now?”
- Match their energy: if they’re casual, be casual. If they’re upset, be gentle. If they want advice, give it directly — don’t hedge forever.
- Use contractions, casual language (“yeah”, “that’s rough”, “I hear you”) — never sound like a textbook.
- End almost every message with ONE specific question or offer, not a generic “how are you feeling?”

Examples of the tone you should use:
- “Oh that sounds really stressful 💜 what’s been piling up?”
- “Exams — yeah that’s a lot of pressure. Do you want to work on a study plan together, or just vent first?”
- “I’m sorry you’re going through this. Is it more the situation itself, or how it’s making you feel about yourself?”

Critical rules:
- NEVER dismiss, minimize or ignore feelings. Always validate first.
- If the user mentions crisis (self-harm, suicide, wanting to disappear), acknowledge with full compassion and provide the crisis line number immediately.
- Every 5–6 exchanges, gently remind them you’re a supportive bot — a real professional can go deeper with them.
- Never ask for personal identifying information.

{emotion_context}
{country_context}
"""

SYSTEM_PROMPT_DAR = """Nta SOUL-INK — sahib/a dyal l9alb, machi t'bib. Tkellm b Darija maghribiya (Franco aw 3arabiya) — kif ma bghiti.
Fahm Franco-Darija (3=3ain, 7=ha, 9=qaf, kh, gh, ch...).

⚠️ MACHI t'bib: Ma t9dresh tdiagnozi wela tdawi. Ila l'app wqfat, 9ul l'bandan y3ayyet l numero dyal azma li t3ta lih.

Nhej dyal l7dit — had shi muhimm bzzaf:
- Jawabat qa9ira w d'fia: 2-3 jumal bss ma lm ykhtar l'bandan akthar.
- MA t9ders tbqa t9ader lmasha3er whdek. Jaweb 3la li gal, w su'al wa7ed m7ded bash tjib akthar.
- Bda b t9bel lmasha3er awwalan ("ah, had shi sa3ib"), w b3d su'al ("ash wqe3?").
- Ila l'bandan 3tak context ("examens", "mmati", "khdma") — dir 7aja m3a: "wah, had shi sa3ib — wash tbghi n3awduk 3la plan? wela bghiti ghir t7ki?"
- Kwi ma bghaw: ila bghaw nsi7a — 3tiha bwed7. Ila bghaw ysiftw — sme3hom. Kwi bwed7.
- Darija 3adiya, machi klinigi: "ah wah", "had shi ta9il", "9bilt fik", "ash 9aderk ndir m3ak?"
- Kmel b su'al wa7ed m7ded — mashi "kif ka t7es?" 3amma.

Amthela:
- "Ah, mashi mezyan had shi 💜 — ash li ka y9elqek bzzaf?"
- "Examens — wah, had shi t9il. Wash tbghi dir plan dyal drasa, wela bghiti ghir t7ki?"
- "Sme3tik. 9bilt fik. Ash l'harder part — lwa9t, wela l'khawf men nnataij?"

Qwa3ed muhimma:
- Ila 3bar 3an afkar dyal kh tar — jaweb b 7nana w 3tih numero dyal azma f dak liha.
- Kul 5-6 risail — fkker lbandan b lwi7a belli nta bot w machi badal 3la mutakhassis.

{emotion_context}
{country_context}
"""

DISCLAIMER     = "SOUL-INK is not therapy or medical advice. If you feel unsafe, please contact a crisis line or emergency services."
DISCLAIMER_DAR = "SOUL-INK machi t'bib wela nsia7a tibbiya. Ila kenti f kh tar, 3ayyet l kh toot azma wella urgences."
DISCLAIMER_AR  = "SOUL-INK ليس معالجًا نفسيًا أو نصيحة طبية. إذا كنت في خطر، اتصل بخط أزمة أو بالطوارئ فورًا."
DISCLAIMER_FR  = "SOUL-INK n'est pas un thérapeute ni un avis médical. Si tu es en danger, contacte une ligne de crise ou les urgences."


EMOTION_COLORS = {
    "joy": "#86efac", "sadness": "#93c5fd", "anger": "#fca5a5",
    "fear": "#fde68a", "shame": "#f9a8d4", "neutral": "#c4b5fd",
}

EMO_FALLBACK = {
    "joy":     "Oh that's genuinely lovely 💜 What's making you feel this way — tell me more?",
    "sadness": "I'm sorry you're feeling this way 💜 What happened?",
    "anger":   "That sounds really frustrating 💜 What's going on?",
    "fear":    "That sounds scary 💜 What's worrying you the most right now?",
    "shame":   "Thank you for sharing that with me 💜 What happened — can you tell me a bit more?",
    "neutral": "I'm here and I'm listening 💜 What's on your mind?",
}

EMO_FALLBACK_DAR = {
    "joy":     "Hamdoullah, had shi zwine 💜 Ash li kayfar7ek haka — 9olha liya?",
    "sadness": "Sme3tik, w rani m3ak 💜 Ash wqe3?",
    "anger":   "Wah, had shi m9arfen 💜 Ash li kayt9el 3lik?",
    "fear":    "Normal t7es haka 💜 Ash kaykh awfek bzzaf daba?",
    "shame":   "Shokran 3la sti9atek fiya 💜 Ash wqe3 — t9der t9olliya shwiya?",
    "neutral": "Rani hna, ka nsme3ek 💜 Ash kayt3eddar 3la ra'sek?",
}

EMO_FALLBACK_AR = {
    "joy":     "الحمد لله! ما الذي يجعلك تشعر بهذا — حدثني أكثر 💜",
    "sadness": "آسف لأنك تمر بهذا 💜 ماذا حدث؟",
    "anger":   "هذا يبدو محبطًا جدًا 💜 ماذا يجري؟",
    "fear":    "هذا منطقي تمامًا 💜 ما الذي يقلقك أكثر الآن؟",
    "shame":   "شكرًا لثقتك بي 💜 ماذا حدث — هل تريد أن تحدثني أكثر؟",
    "neutral": "أنا هنا وأسمعك 💜 ما الذي دار في ذهنك؟",
}

EMO_FALLBACK_FR = {
    "joy":     "Oh c'est vraiment bien 💜 Qu'est-ce qui te rend si heureux/heureuse — dis-m'en plus?",
    "sadness": "Je suis désolé/e que tu traverses ça 💜 Qu'est-ce qui s'est passé?",
    "anger":   "Ça semble vraiment frustrant 💜 Qu'est-ce qui se passe?",
    "fear":    "Je comprends que ce soit angoissant 💜 Qu'est-ce qui t'inquiète le plus en ce moment?",
    "shame":   "Merci de me faire confiance 💜 Qu'est-ce qui s'est passé — tu peux m'en dire un peu plus?",
    "neutral": "Je suis là et je t'écoute 💜 Qu'est-ce qui te passe par la tête?",
}

# ── Arabic system prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT_AR = """أنت SOUL-INK — صديق حقيقي يستمع، ليس طبيبًا. تحدث بالعربية الفصحى الطبيعية الدافئة.

⚠️ لست معالجًا نفسيًا أو طبيبًا. إذا توقف التطبيق، أخبر المستخدم بالاتصال بخط الأزمة المحلي.

أسلوب المحادثة — هذا مهم جدًا:
- ردود قصيرة ودافئة وطبيعية: 2-3 جمل كحد أقصى ما لم يحتج المستخدم أكثر.
- لا تنتظر حتى يقول كل شيء. تجاوب مع ما شاركه، ثم اسأل سؤالاً واحدًا محددًا.
- اعترف بالشعور أولاً ("آسف لأنك تمر بهذا")  ثم اسأل ("ماذا حدث؟").
- إذا أعطاك السياق ("امتحانات"، "عمل"، "عائلة") — انتقل فورًا إلى شيء عملي: "هذا صعب — هل تريد أن نضع خطة معًا أو تفضفض فقط؟"
- اتبع طاقتهم: إذا كانوا يريدون نصيحة — أعطها مباشرة. إذا أرادوا التنفيس — اسمع.
- انتهِ كل رد بسؤال واحد محدد — ليس "كيف تشعر؟" العامة.

أمثلة:
- "آسف لأنك تمر بهذا 💜 — ماذا حدث؟"
- "امتحانات — هذا ضغط كثير. هل تريد أن نضع خطة دراسݺة او تفضفض أولا؟"
- "ما الجزء الأصعب — الوقت، أوم الخوف من النتائج؟"

قواعد:
- لا تُقلّل من المشاعر أبدًا.
- إذا أشار إلى أزمة — أعطه رقم خط الأزمة فورًا.
- كل 5-6 رسائل ذكّره برفق أنك بوت وليس بديلاً عن متخصص.

{emotion_context}
{country_context}
"""

# ── French system prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT_FR = """Tu es SOUL-INK — un ami bienveillant qui écoute vraiment, pas un thérapeute.
Réponds toujours en français, chaleureusement et naturellement.

⚠️ Tu n'es PAS un thérapeute ou médecin. Si l'appli ne fonctionne plus, dis à l'utilisateur d'appeler sa ligne de crise locale.

Ton style de conversation — c'est crucial :
- Réponses courtes, chaudes et naturelles : 2-3 phrases max sauf si l'utilisateur en a besoin de plus.
- N'attends PAS qu'il te donne tout. Réagis à ce qu'il dit, puis pose UNE question spécifique pour en savoir plus.
- Commence par valider la émotion ("oh c'est vraiment dur"), puis demande ce qui s'est passé ("qu'est-ce qui s'est passé ?").
- Quand il te donne du contexte ("les examens", "ma mère", "le boulot") — rebondis directement : "ouais c'est lourd — t'as envie qu'on voit ça ensemble ou tu veux juste parler d'abord ?"
- Si l'utilisateur veut un conseil, donne-le directement. S'il veut juste parler, écoute.
- Parle naturellement : "ouais", "c'est dur", "je t'entends", "t'as raison" — jamais comme un manuel.
- Termine presque chaque message par UNE question ou proposition spécifique.

Exemples du ton voulu :
- "Oh c'est vraiment stressant 💜 — qu'est-ce qui pile le plus en ce moment ?"
- "Les examens — ouais c'est une grosse pression. T'as envie qu'on fasse un plan de révisions ensemble, ou tu veux juste véntiler d'abord ?"
- "Je suis désolé/e que tu traverses ça. C'est plutôt la situation elle-même, ou ce qu'elle te fait ressentir sur toi ?"

Règles importantes :
- Ne minimise jamais les émotions. Valide toujours en premier.
- Si l'utilisateur mentionne une crise, réponds avec compassion et donne le numéro de ligne de crise immédiatement.
- Toutes les 5-6 échanges, rappelle doucement que tu es un bot de soutien, pas un remplacement pour un vrai professionnel.

{emotion_context}
{country_context}
"""


# ── Language UI strings ───────────────────────────────────────────────────────
LANG = {
    "en": {
        "language_header": "### 🌍 Language",
        "local_ai_header": "### 🤖 Local AI",
        "session_header":  "### 📊 This Session",
        "privacy_header":  "### 🔒 Privacy",
        "messages_lbl":    "Messages",
        "duration_lbl":    "Duration",
        "mood_lbl":        "**Mood this session**",
        "themes_lbl":      "**Themes**",
        "export_btn":      "Export chat (.json)",
        "delete_btn":      "🗑 Delete session & start over",
        "download_btn":    "⬇ Download",
        "placeholder":     "Say anything... I'm listening 💜",
        "status_on":       "Ollama Online",
        "status_off":      "Ollama Offline",
        "model_lbl":       "Model",
        "model_help":      "Choose from your locally pulled Ollama models.",
        "footer":          "💜 Session-only. Nothing is saved when you leave.<br>Powered by Ollama · GoEmotions · spaCy · 100% local",
        "disclaimer":      DISCLAIMER,
        "breathe_circle":  "Breathe<br>in · hold<br>out · hold",
        "breathe_desc":    "4 counts in → 4 hold → 4 out → 4 hold<br>Repeat for 2 minutes 💜",
        "breathe_reply":   "Of course 💜 Here's a breathing exercise — let it guide you. Breathe in for 4 counts, hold for 4, breathe out for 4, hold for 4. Repeat as long as you need. How are you feeling right now?",
        "ground_title":    "🌿 5-4-3-2-1 Grounding Exercise",
        "ground_body":     "Take a slow breath, then notice:\n\n**👁 5 things you can SEE** — look around and name them  \n**✋ 4 things you can TOUCH** — feel their texture  \n**👂 3 things you can HEAR** — listen carefully  \n**👃 2 things you can SMELL** — or imagine your favourite scent  \n**👅 1 thing you can TASTE** — or notice the sensation in your mouth  \n\nThis brings you back to the present moment. You're safe right here. 💜",
        "ground_reply":    "Let's ground you 💜 I've opened the 5-4-3-2-1 exercise below. Take your time with each step — there's no rush. When you're ready, come back and tell me how you feel.",
        "offline_warn":    "🤖 **Ollama is not running.** Showing a supportive reply now.\n\nTo enable full AI: install [Ollama](https://ollama.com/download), run `ollama pull llama3` and `ollama serve`, then refresh.",
        "crisis_live":     "💜 I can hear how much pain you're in right now, and I'm so grateful you're still here and talking. **You are not alone.** Please reach out to a crisis line — a real person is there for you right now:",
        "crisis_fall":     "💜 I hear you, and I'm so glad you're talking about this rather than holding it alone. **Please reach out to a crisis line right now — they are there exactly for moments like this:**",
        "fallback":        EMO_FALLBACK,
        "system_prompt":   SYSTEM_PROMPT,
        "main_title":      "# SOUL-INK — The AI That Listens",
        "main_sub":        "<p style='color:#a78bba;font-size:0.92rem;margin-top:-0.5rem;margin-bottom:1rem'>💜 Anonymous · Session-only · 100% local · No data stored when you leave</p>",
        "welcome":         "Hey, I'm really glad you're here. 💜\n\nThis is a safe, judgment-free space. You can tell me anything — what's been weighing on you, how your day went, something you can't stop thinking about. Or if you're not sure where to start, that's okay too.\n\n**A few things you can try:**\n- Just talk — I'll listen and respond with care\n- Type `/breathe` for a guided breathing exercise\n- Type `/ground` for a grounding exercise when things feel overwhelming\n\nWhat's on your mind today?",
        "install_guide":   '<div style="font-size:0.82rem;color:#9d87b0;margin-top:0.6rem;line-height:1.6"><b>To start the AI:</b><br>1. Install <a href="https://ollama.com/download" target="_blank" style="color:#c084fc">ollama.com</a><br>2. Run in a terminal:<br><code style="color:#c084fc">ollama pull llama3</code><br><code style="color:#c084fc">ollama serve</code><br>3. Refresh this page 💜</div>',
        "arch_title":      "🧠 How it works — NLP pipeline",
        "arch_steps": [
            {"n": 1, "title": "Language detection", "desc": "Identifies Arabic, Darija (Franco or Arabic script), French, or English from your message."},
            {"n": 2, "title": "Emotion detection", "desc": "A DistilBERT model fine-tuned on GoEmotions classifies the emotional tone (joy, sadness, anger, fear, shame...)."},
            {"n": 3, "title": "Theme extraction", "desc": "spaCy pulls out key topics and recurring keywords from the conversation."},
            {"n": 4, "title": "Crisis detection", "desc": "Scans for risk phrases (EN/FR/AR/Darija) using a 70+ country helpline database and the DODa dataset."},
            {"n": 5, "title": "Response generation", "desc": "A local LLM (Ollama — Llama3, Gemma3, Qwen3, Mistral...) generates an empathetic reply guided by the emotional context."},
        ],
        "arch_footer":     "Everything runs locally. No data is ever sent or stored.",
    },
    "dar": {
        "language_header": "### 🌍 l'Logha",
        "local_ai_header": "### 🤖 Dka' Mahalli",
        "session_header":  "### 📊 Had ljalsa",
        "privacy_header":  "### 🔒 l'Khsousia",
        "messages_lbl":    "Risail",
        "duration_lbl":    "l'Mudda",
        "mood_lbl":        "**l'7al f had ljalsa**",
        "themes_lbl":      "**lMawadi3**",
        "export_btn":      "Exporty l7dit (.json)",
        "delete_btn":      "🗑 Msah l7dit w bda men jdid",
        "download_btn":    "⬇ Nzel",
        "placeholder":     "9ol chay... ka nsme3ek 💜",
        "status_on":       "Ollama متصل",
        "status_off":      "Ollama غير متصل",
        "model_lbl":       "lModel",
        "model_help":      "Khttar men lmodels li 3endek f Ollama.",
        "footer":          "💜 Ljalsa ghir dyal had lwa9t. Walo kaytkhzen mnin tmshi.<br>Powered by Ollama · GoEmotions · spaCy · 100% ma7elli",
        "disclaimer":      DISCLAIMER_DAR,
        "breathe_circle":  "Tnfs<br>dkhel · t9ella<br>khrrej · t9ella",
        "breathe_desc":    "4 dkhel → 4 t9ella → 4 khrrej → 4 t9ella<br>3awwd l'muddat 2 d9i9a 💜",
        "breathe_reply":   "Wakhkha 💜 Ha tamrin dyal tnfs — khallieh ywjhek. Tnfs dkhel 4 lhzat, t9ella 4, ttnfs khrrej 4, t9ella 4. 3awwd ma bghiti. Kif ka t7es daba?",
        "ground_title":    "🌿 Tamrin 5-4-3-2-1 dyal T'assis",
        "ground_body":     "Khud tnfs bwahd, w b3d l7wed:\n\n**👁 5 7wayej t9der tshufhom** — shuf dawrek w summihom  \n**✋ 4 7wayej t9der tmsshom** — 7es nwi3athom  \n**👂 3 7wayej t9der tsme3hom** — sme3 mezyan  \n**👃 2 7wayej t9der tshemhom** — wella ttsawwar riha li t7ebbha  \n**👅 7aja wa7da t9der tdawqha** — wella l7wed f temk  \n\nHada ghadi yrje3ek l'had llahda. Nta/Nti salim/a hna. 💜",
        "ground_reply":    "Yalla nt'assaw 💜 Fetht lik tamrin 5-4-3-2-1 l'te7t. Dir wa7ed wa7ed — mafich 3jala. Mnin tkammel, ji 9olliya kif t7esti.",
        "offline_warn":    "🤖 **Ollama mashi shaghel.** Ka njawbek b jawab dyal d3em daba.\n\nBash t5eddem l'AI l-kamel: nseb [Ollama](https://ollama.com/download), w dir `ollama pull llama3` w `ollama serve`, w b3d refreshy.",
        "crisis_live":     "💜 Ka nsme3ek w ka n7es b alam li 3endek, w rani ferhan/a billi nta/nti mazal hna w katat7emma. **Nta/Nti mawahed/a.** 3ayyet l kh toot azma — kayn shi 7ad ghadi ysme3ek f had llahda:",
        "crisis_fall":     "💜 Sme3tik, w rani ferhan/a billi tkellmti 3wad ma tbqa wahed/a m3a had shi. **3ayyet l kh toot azma daba — kaynin 3endek ghir f had llahda:**",
        "fallback":        EMO_FALLBACK_DAR,
        "system_prompt":   SYSTEM_PROMPT_DAR,
        "main_title":      "# SOUL-INK — يسمعك",
        "main_sub":        "<p style='color:#a78bba;font-size:0.92rem;margin-top:-0.5rem;margin-bottom:1rem'>💜 Anonimi · Ljalsa bas · 100% ma7elli · Walo kaytkhzen mnin tmshi</p>",
        "welcome":         "Merhba bik/biki, rani ferhan/a billi jiti. 💜\n\nHad lmakan amin w bla 7okm. T9der t9ol li kull shi — li kayt9elek, kif mrat yumek, chi 7aja ma tzalsh kayfker fiha. Aw ila ma3refshi men fin tbda, hada mazyan.\n\n**Chi 7wayej t9der tjerbhom:**\n- Hdr ghir — ghadi nsme3ek w njawbek b 7nana\n- Kteb `/breathe` bash n3tik tamrin dyal tnfs\n- Kteb `/ground` bash n3tik tamrin dyal t'assis\n\nAsh kayt3eddar 3la ra'sek lyum?",
        "install_guide":   '<div style="font-size:0.82rem;color:#9d87b0;margin-top:0.6rem;line-height:1.6"><b>Bash tbda l\'AI:</b><br>1. Nseb <a href="https://ollama.com/download" target="_blank" style="color:#c084fc">ollama.com</a><br>2. Dir f terminal:<br><code style="color:#c084fc">ollama pull llama3</code><br><code style="color:#c084fc">ollama serve</code><br>3. Refreshy had lpa9a 💜</div>',
        "arch_title":      "🧠 Kifash khdama — pipeline NLP",
        "arch_steps": [
            {"n": 1, "title": "Kashf l'Logha", "desc": "Ka y3ref ila kteb b 3arabiya, darija (franco wella 3arabiya), fransawiya wella nglizia."},
            {"n": 2, "title": "Kashf l'Mash3our", "desc": "Model DistilBERT mdarrab 3la GoEmotions ka y3ref l'7ala l'wijdaniya (far7, 7zen, ghadab, khouf...)."},
            {"n": 3, "title": "Khrouj l'Mawadi3", "desc": "spaCy ka y9ess l'mawadi3 w lkalimat l'muhimma f l7dit."},
            {"n": 4, "title": "Kashf l'Azma", "desc": "Tqelleb 3la jumlat dyal khtar (3arabiya/darija/fransawiya/nglizia) b base dyal +70 blad w DODa."},
            {"n": 5, "title": "Generation dyal Jawab", "desc": "Model AI ma7elli (Ollama — Llama3, Gemma3, Qwen3, Mistral...) ka yjawb b ta3awoun w fhem l'7ala dyalek."},
        ],
        "arch_footer":     "Kullshi khdama ma7elliyan. Walo katsift wla katkhzen.",
    },
    "ar": {
        "language_header": "### 🌍 اللغة",
        "local_ai_header": "### 🤖 الذكاء الاصطناعي المحلي",
        "session_header":  "### 📊 هذه الجلسة",
        "privacy_header":  "### 🔒 الخصوصية",
        "messages_lbl":    "الرسائل",
        "duration_lbl":    "المدة",
        "mood_lbl":        "**المزاج خلال الجلسة**",
        "themes_lbl":      "**المواضيع**",
        "export_btn":      "تصدير المحادثة (.json)",
        "delete_btn":      "🗑 حذف الجلسة والبدء من جديد",
        "download_btn":    "⬇ تنزيل",
        "placeholder":     "قل ما تريد... أنا هنا أسمعك 💜",
        "status_on":       "Ollama متصل",
        "status_off":      "Ollama غير متصل",
        "model_lbl":       "النموذج",
        "model_help":      "اختر من النماذج المثبتة محليًا في Ollama.",
        "footer":          "💜 جلسة فقط. لا شيء يُحفظ عند مغادرتك.<br>مدعوم بـ Ollama · GoEmotions · spaCy · 100% محلي",
        "disclaimer":      DISCLAIMER_AR,
        "breathe_circle":  "تنفس<br>داخل · توقف<br>خارج · توقف",
        "breathe_desc":    "4 داخل → 4 توقف → 4 خارج → 4 توقف<br>كرر لمدة دقيقتين 💜",
        "breathe_reply":   "بالطبع 💜 إليك تمرين تنفس — دعه يرشدك. تنفس للداخل 4 عدات، احتفظ 4، أخرج 4، احتفظ 4. كرر ما تحتاج. كيف تشعر الآن؟",
        "ground_title":    "🌿 تمرين التأريض 5-4-3-2-1",
        "ground_body":     "خذ نفسًا عميقًا ببطء، ثم لاحظ:\n\n**👁 5 أشياء تراها** — انظر حولك وسمّها\n**✋ 4 أشياء تلمسها** — اشعر بملمسها\n**👂 3 أشياء تسمعها** — استمع جيدًا\n**👃 شيئان تشمّهما** — أو تخيّل رائحة محببة\n**👅 شيء واحد تتذوقه** — أو لاحظ الإحساس في فمك\n\nهذا يعيدك إلى اللحظة الحاضرة. أنت بأمان هنا. 💜",
        "ground_reply":    "لنتأرض معًا 💜 لقد فتحت لك تمرين 5-4-3-2-1 أدناه. خذ وقتك مع كل خطوة — لا تسرع. عندما تنتهي، أخبرني كيف تشعر.",
        "offline_warn":    "🤖 **Ollama لا يعمل.** سأعرض عليك ردًا داعمًا الآن.\n\nلتفعيل الذكاء الاصطناعي الكامل: ثبّت [Ollama](https://ollama.com/download)، ثم نفّذ `ollama pull llama3` و`ollama serve`، وأعد تحميل الصفحة.",
        "crisis_live":     "💜 أسمعك وأشعر بحجم الألم الذي تعانيه، وأنا سعيد/ة لأنك لا تزال هنا وتتحدث. **لست وحدك/ي.** يرجى التواصل مع خط الأزمة — شخص حقيقي موجود لك الآن:",
        "crisis_fall":     "💜 أسمعك، وأنا سعيد/ة لأنك تتحدث بدلاً من أن تواجه هذا وحدك. **يرجى الاتصال بخط الأزمة الآن — هم موجودون تحديدًا في مثل هذه اللحظات:**",
        "fallback":        EMO_FALLBACK_AR,
        "system_prompt":   SYSTEM_PROMPT_AR,
        "main_title":      "# SOUL-INK — الذكاء الذي يسمعك",
        "main_sub":        "<p style='color:#a78bba;font-size:0.92rem;margin-top:-0.5rem;margin-bottom:1rem;direction:rtl'>💜 مجهول الهوية · جلسة فقط · 100% محلي · لا يُحفظ شيء عند مغادرتك</p>",
        "welcome":         "أهلًا وسهلًا، يسعدني وجودك هنا. 💜\n\nهذا مكان آمن وخالٍ من الأحكام. يمكنك أن تحدثني عن أي شيء — ما يُثقلك، كيف كان يومك، شيء لا يفارق تفكيرك. أو إذا لم تعرف من أين تبدأ، فهذا جيد أيضًا.\n\n**أشياء يمكنك تجربتها:**\n- تحدث فقط — سأستمع وأرد باهتمام\n- اكتب `/breathe` لتمرين تنفس موجّه\n- اكتب `/ground` لتمرين التأريض عند الإرهاق\n\nما الذي يدور في ذهنك اليوم؟",
        "install_guide":   '<div style="font-size:0.82rem;color:#9d87b0;margin-top:0.6rem;line-height:1.9;direction:rtl"><b>لتشغيل الذكاء الاصطناعي:</b><br>1. ثبّت <a href="https://ollama.com/download" target="_blank" style="color:#c084fc">ollama.com</a><br>2. نفّذ في الطرفية:<br><code style="color:#c084fc">ollama pull llama3</code><br><code style="color:#c084fc">ollama serve</code><br>3. أعد تحميل الصفحة 💜</div>',
        "arch_title":      "🧠 كيف يعمل — خط معالجة اللغة الطبيعية",
        "arch_steps": [
            {"n": 1, "title": "كشف اللغة", "desc": "يحدد العربية الفصحى، الدارجة (بالحروف اللاتينية أو العربية)، الفرنسية أو الإنجليزية من رسالتك."},
            {"n": 2, "title": "كشف المشاعر", "desc": "نموذج DistilBERT مدرّب على GoEmotions يصنّف النغمة العاطفية (فرح، حزن، غضب، خوف، خجل...)."},
            {"n": 3, "title": "استخراج المواضيع", "desc": "مكتبة spaCy تحدد المواضيع والكلمات المفتاحية المتكررة في المحادثة."},
            {"n": 4, "title": "كشف الأزمات", "desc": "بحث عن عبارات الخطر (عربية/دارجة/فرنسية/إنجليزية) عبر قاعدة بيانات لأكثر من 70 دولة وقاموس DODa."},
            {"n": 5, "title": "توليد الردّ", "desc": "نموذج ذكاء اصطناعي محلي (Ollama — Llama3, Gemma3, Qwen3, Mistral...) يولّد ردًا متعاطفًا مبنيًا على السياق العاطفي."},
        ],
        "arch_footer":     "كل شيء يعمل محليًا. لا تُرسل أو تُخزَّن أي بيانات.",
    },
    "fr": {
        "language_header": "### 🌍 Langue",
        "local_ai_header": "### 🤖 IA Locale",
        "session_header":  "### 📊 Cette session",
        "privacy_header":  "### 🔒 Confidentialité",
        "messages_lbl":    "Messages",
        "duration_lbl":    "Durée",
        "mood_lbl":        "**Humeur cette session**",
        "themes_lbl":      "**Thèmes**",
        "export_btn":      "Exporter la conversation (.json)",
        "delete_btn":      "🗑 Supprimer et recommencer",
        "download_btn":    "⬇ Télécharger",
        "placeholder":     "Dis-moi tout... je t'écoute 💜",
        "status_on":       "Ollama connecté",
        "status_off":      "Ollama hors-ligne",
        "model_lbl":       "Modèle",
        "model_help":      "Choisis parmi tes modèles Ollama installés localement.",
        "footer":          "💜 Session uniquement. Rien n'est sauvegardé à la fermeture.<br>Propulsé par Ollama · GoEmotions · spaCy · 100% local",
        "disclaimer":      DISCLAIMER_FR,
        "breathe_circle":  "Respire<br>inspire · tiens<br>expire · tiens",
        "breathe_desc":    "4 inspire → 4 bloque → 4 expire → 4 bloque<br>Répète 2 minutes 💜",
        "breathe_reply":   "Bien sûr 💜 Voici un exercice de respiration — laisse-le te guider. Inspire 4 temps, bloque 4, expire 4, bloque 4. Répète autant que nécessaire. Comment tu te sens là ?",
        "ground_title":    "🌿 Exercice d'ancrage 5-4-3-2-1",
        "ground_body":     "Prends une grande inspiration, puis remarque:\n\n**👁 5 choses que tu VOIS** — regarde autour et nomme-les  \n**✋ 4 choses que tu TOUCHES** — sens leur texture  \n**👂 3 choses que tu ENTENDS** — écoute attentivement  \n**👃 2 choses que tu SENS** — ou imagine ton odeur préférée  \n**👅 1 chose que tu GOÛTES** — ou remarque la sensation dans ta bouche  \n\nCa te ramène au moment présent. Tu es en sécurité ici. 💜",
        "ground_reply":    "On va s'ancrer ensemble 💜 J'ai ouvert l'exercice 5-4-3-2-1 en dessous. Prends ton temps — il n'y a pas de presse. Quand tu es prêt/e, dis-moi comment tu te sens.",
        "offline_warn":    "🤖 **Ollama ne tourne pas.** Je te montre une réponse de soutien pour l'instant.\n\nPour activer l'IA complète : installe [Ollama](https://ollama.com/download), lance `ollama pull llama3` et `ollama serve`, puis rafraîchis.",
        "crisis_live":     "💜 Je t'entends et je sens à quel point tu souffres. Je suis soulagé/e que tu sois encore là et que tu parles. **Tu n'es pas seul/e.** Contacte une ligne de crise — une vraie personne est là pour toi maintenant :",
        "crisis_fall":     "💜 Je t'entends, et je suis content/e que tu en parles plutôt que de le garder seul/e. **Appelle une ligne de crise maintenant — ils sont là exactement pour ces moments :**",
        "fallback":        EMO_FALLBACK_FR,
        "system_prompt":   SYSTEM_PROMPT_FR,
        "main_title":      "# SOUL-INK — L'IA qui t'écoute",
        "main_sub":        "<p style='color:#a78bba;font-size:0.92rem;margin-top:-0.5rem;margin-bottom:1rem'>💜 Anonyme · Session uniquement · 100% local · Rien n'est sauvegardé</p>",
        "welcome":         "Hey, je suis vraiment content/e que tu sois là. 💜\n\nC'est un espace sûr et sans jugement. Tu peux me parler de tout — ce qui te pèse, comment s'est passée ta journée, quelque chose qui ne te quitte pas. Ou si tu sais pas par où commencer, c'est aussi okay.\n\n**Quelques choses que tu peux essayer :**\n- Parle simplement — je t'écoute et je réponds avec soin\n- Tape `/breathe` pour un exercice de respiration guidé\n- Tape `/ground` pour un exercice d'ancrage quand ça devient trop\n\nQu'est-ce qui se passe pour toi aujourd'hui ?",
        "install_guide":   '<div style="font-size:0.82rem;color:#9d87b0;margin-top:0.6rem;line-height:1.6"><b>Pour lancer l\'IA :</b><br>1. Installe <a href="https://ollama.com/download" target="_blank" style="color:#c084fc">ollama.com</a><br>2. Lance dans un terminal :<br><code style="color:#c084fc">ollama pull llama3</code><br><code style="color:#c084fc">ollama serve</code><br>3. Rafraîchis la page 💜</div>',
        "arch_title":      "🧠 Comment ça marche — pipeline NLP",
        "arch_steps": [
            {"n": 1, "title": "Détection de la langue", "desc": "Identifie l'arabe, le darija (franco ou en arabe), le français ou l'anglais à partir de ton message."},
            {"n": 2, "title": "Détection d'émotion", "desc": "Un modèle DistilBERT affiné sur GoEmotions classe le ton émotionnel (joie, tristesse, colère, peur, honte...)."},
            {"n": 3, "title": "Extraction de thèmes", "desc": "spaCy repère les sujets clés et mots-clés récurrents de la conversation."},
            {"n": 4, "title": "Détection de crise", "desc": "Recherche de phrases à risque (FR/EN/AR/Darija) via une base de 70+ pays et le dataset DODa."},
            {"n": 5, "title": "Génération de réponse", "desc": "Un LLM local (Ollama — Llama3, Gemma3, Qwen3, Mistral...) génère une réponse empathique, guidée par le contexte émotionnel."},
        ],
        "arch_footer":     "Tout tourne en local. Aucune donnée n'est envoyée ni stockée.",
    },
}

def S():
    """Return the LANG strings dict for the current session language."""
    return LANG.get(st.session_state.get("language", "en"), LANG["en"])

def detect_darija_arabic(text: str) -> bool:
    """True if text contains Arabic-script characters (any Arabic Unicode block)."""
    return any('\u0600' <= c <= '\u06ff' for c in text)


def detect_darija_franco(text: str) -> bool:
    """
    True if text looks like Franco-Darija (Arabizi written in Latin characters).
    Heuristic: checks for characteristic letter+digit combos (3, 7, 9, 2)
    combined with Darija vocabulary markers.
    """
    t = text.lower()
    # Arabizi digit markers: 3 (ain), 7 (ha/7a), 9 (qaf), 2 (hamza)
    has_digit_marker = bool(re.search(r'\b\w*[379]\w*\b', t))
    # Common Franco-Darija words / particles used very frequently
    darija_lexicon = {
        "wah", "ewa", "mashi", "bzzaf", "bezzaf", "bghit", "kayn", "kayna",
        "hna", "nta", "nti", "ntuma", "rani", "radi", "daba", "dyal",
        "ghadi", "3ndek", "3ndi", "3lik", "3la", "f had", "had shi",
        "mezyan", "mzyan", "zwina", "safi", "khali", "khouya", "khti",
        "mhemm", "wakha", "wakhkha", "yallah", "hamdoullah", "inchallah",
        "chwiya", "shwiya", "w rani", "w nta", "w nti", "b7al", "b7alha",
        "tkhelli", "tkhalli", "nsme3ek", "njawbek", "ndiru", "ndir",
        "kteb", "ktab", "l7dit", "l7al", "lhall", "lmkan", "lmakan",
        "tnfs", "t9ella", "tamrin", "ljalsa", "lmasha3er", "lmsha3er",
        "khssni", "khassni", "bghit nmout", "3ayyet", "ferhan", "fer7an",
    }
    has_darija_word = any(w in t for w in darija_lexicon)
    # Digraph combos common in Darija romanisation
    has_digraph = bool(re.search(r'\b(kh|gh|ch|sh|ts|dj|nd)\w', t))
    # At least digit marker OR (darija word AND digraph)
    return has_digit_marker or (has_darija_word and has_digraph) or (sum([has_digit_marker, has_darija_word, has_digraph]) >= 2)


def detect_darija(text: str) -> bool:
    """True if text is in Darija — either Arabic script or Franco-Darija Latin."""
    return detect_darija_arabic(text) or detect_darija_franco(text)


def detect_french(text: str) -> bool:
    """
    True if text is likely French.
    Heuristic: French-specific accented characters + common French function words.
    """
    t = text.lower()
    # French-specific accented chars (not shared with Spanish/Portuguese as heavily)
    has_fr_chars = bool(re.search(r'[éèêëàâîïôùûüÿœæç]', t))
    # Common French-only words / particles
    fr_lexicon = {
        "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
        "est", "sont", "c'est", "n'est", "pas", "dans", "avec", "sur",
        "pour", "ça", "ca", "oui", "non", "mais", "donc", "comme",
        "très", "bien", "beaucoup", "merci", "aussi", "toujours",
        "jamais", "encore", "parce", "vraiment", "quand", "comment",
        "qu'est", "j'ai", "j'me", "t'as", "c'était", "ouais",
        "trop", "grave", "genre", "truc", "chose", "aujourd",
    }
    has_fr_word = any(w in t.split() or w in t for w in fr_lexicon)
    return has_fr_chars or (has_fr_word and len(text.split()) >= 2)



# ── Ollama helpers ────────────────────────────────────────────────────────────
def check_ollama_status() -> bool:
    """Return True if the local Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def get_ollama_models() -> list:
    """Return list of locally pulled model names from Ollama."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        if r.status_code == 200:
            data = r.json()
            names = [m["name"].split(":")[0] for m in data.get("models", [])]
            return list(dict.fromkeys(names)) or AVAILABLE_MODELS
    except Exception:
        pass
    return AVAILABLE_MODELS

def call_ollama(model: str, messages: list, system_prompt: str) -> str:
    """
    Send conversation history to Ollama's /api/chat endpoint.
    Uses streaming=False for simplicity.
    """
    payload_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = "user" if m["role"] == "user" else "assistant"
        payload_messages.append({"role": role, "content": m["content"]})

    payload = {
        "model": model,
        "messages": payload_messages,
        "stream": False,
        "options": {"temperature": 0.85, "num_predict": 700},
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"].strip()

# ── Lazy local model loaders ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_emotion_pipe():
    try:
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   TextClassificationPipeline)
        model = "joeddav/distilbert-base-uncased-go-emotions"
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForSequenceClassification.from_pretrained(model)
        pipe = TextClassificationPipeline(
            model=mdl, tokenizer=tok,
            return_all_scores=True, function_to_apply="sigmoid"
        )
        return pipe, mdl.config.id2label
    except Exception:
        return None, None

@st.cache_resource(show_spinner=False)
def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None

# ── NLP helpers ───────────────────────────────────────────────────────────────
def detect_crisis(text: str) -> bool:
    t = text.lower()
    if any(phrase in t for phrase in CRISIS_PHRASES_EN):
        return True
    # Also check the DODa-enriched Darija phrase set (cached)
    darija_phrases = _load_doda_crisis_terms()
    return any(phrase in t for phrase in darija_phrases)


def detect_emotion(text: str):
    pipe, _ = _load_emotion_pipe()
    if pipe is None:
        return "neutral", {}
    try:
        raw = pipe(text[:512])[0]
        grouped = {g: 0.0 for g in EMO_GROUPS}
        for s in raw:
            lab, sc = s["label"], float(s["score"])
            for g, members in EMO_GROUPS.items():
                if lab in members and sc > grouped[g]:
                    grouped[g] = sc
        keep = {k: round(v, 3) for k, v in grouped.items() if v >= 0.28}
        coarse = keep or {k: round(v, 3) for k, v in
                          sorted(grouped.items(), key=lambda x: x[1], reverse=True)[:2]}
        primary = max(coarse, key=lambda k: coarse[k])
        return primary, coarse
    except Exception:
        return "neutral", {}

def extract_triggers(text: str, max_terms: int = 5) -> list:
    nlp = _load_spacy()
    if nlp is None:
        return []
    try:
        doc = nlp(text[:500])
        chunks = [c.text.strip().lower() for c in doc.noun_chunks if len(c.text.strip()) > 1]
        tokens = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
        common = [w for w, _ in Counter(tokens).most_common(10)]
        seen, out = set(), []
        for x in chunks + common:
            if x not in seen:
                seen.add(x); out.append(x)
            if len(out) >= max_terms:
                break
        return out
    except Exception:
        return []

def build_emotion_context(primary: str, coarse: dict) -> str:
    if not coarse:
        return ""
    scores = ", ".join(f"{k}: {v}" for k, v in coarse.items())
    return f"\n\n[Emotional context — primary: {primary} | signals: {scores}. Use this to be more attuned but do NOT mention these labels to the user.]"

def get_session_duration() -> str:
    secs = int(time.time() - st.session_state.session_start)
    if secs < 60: return f"{secs}s"
    return f"{secs // 60}m {secs % 60}s"

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":       [],
        "session_start":  time.time(),
        "show_breathe":   False,
        "show_ground":    False,
        "emotion_log":    [],
        "ollama_model":   "llama3",
        "language":       "en",
        "user_country":   "Not specified",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
s = S()  # active language strings dict — recalculated each rerun

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem">
  <div style="width:38px;height:38px;border-radius:12px;
              background:linear-gradient(135deg,var(--ink2),var(--ink));
              display:flex;align-items:center;justify-content:center;
              font-size:1.2rem;flex-shrink:0;box-shadow:0 3px 12px var(--glow)">💜</div>
  <div style="font-size:1.25rem;font-weight:700;
              background:linear-gradient(135deg,#e9d5ff,#a855f7);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">SOUL-INK</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(f'<span class="disclaimer">{DISCLAIMER}</span>', unsafe_allow_html=True)

    # ── Language indicator (auto-detected — not a control) ───────────────────
    _lang_badge = {
        "en":  ("🇬🇧", "English"),
        "dar": ("🇲🇦", "Darija"),
        "ar":  ("🌐", "العربية"),
        "fr":  ("🇫🇷", "Français"),
    }.get(st.session_state.language, ("🇬🇧", "English"))
    st.markdown(
        f"<div style='font-size:0.78rem;color:#9d87b0;margin-bottom:0.2rem'>"
        f"🌐 Language — auto-detected from your messages"
        f"</div>"
        f"<div style='background:rgba(168,85,247,0.1);border:1px solid rgba(168,85,247,0.25);"
        f"border-radius:999px;padding:3px 12px;display:inline-block;font-size:0.82rem;"
        f"color:#c084fc;margin-bottom:0.6rem'>{_lang_badge[0]} {_lang_badge[1]}"
        f"<span style='color:#6b5a7a;font-size:0.72rem;margin-left:8px'>type /lang en · fr · dar · ar to override</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Country / helpline selector ─────────────────────────────────────────
    st.markdown("### 🌍 Your Location")
    st.markdown(
        "<div style='font-size:0.82rem;color:#9d87b0;margin-bottom:0.4rem'>"
        "We use this only to show you the right emergency &amp; crisis number."
        "</div>",
        unsafe_allow_html=True,
    )
    _c_idx = COUNTRY_LIST.index(st.session_state.user_country) \
             if st.session_state.user_country in COUNTRY_LIST else len(COUNTRY_LIST) - 1
    _chosen = st.selectbox(
        "Your country",
        options=COUNTRY_LIST,
        index=_c_idx,
        key="country_select",
        help="Used only to display the correct crisis helpline. Nothing is stored.",
    )
    st.session_state.user_country = _chosen

    if _chosen != "Not specified" and _chosen in HELPLINES_DB:
        _hd = HELPLINES_DB[_chosen]
        st.markdown(
            f"<div style='font-size:0.8rem;background:rgba(168,85,247,0.09);"
            f"border:1px solid rgba(168,85,247,0.22);border-radius:10px;"
            f"padding:0.55rem 0.8rem;margin-top:0.3rem;line-height:1.7'>"
            f"{_hd['flag']} <b>Emergency:</b> {_hd['emergency']}<br>"
            f"📞 <b>Crisis line:</b> <b>{_hd['crisis']}</b><br>"
            f"<small style='color:#a78bba'>{_hd['name']}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Ollama status
    st.markdown(s["local_ai_header"])
    online = check_ollama_status()

    if online:
        st.markdown('<span class="status-online"><span class="dot-pulse">●</span> Ollama Online</span>', unsafe_allow_html=True)
        available = get_ollama_models()
        # Ensure current model is in list
        if st.session_state.ollama_model not in available:
            available = [st.session_state.ollama_model] + available
        selected = st.selectbox(
            "Model",
            options=available,
            index=available.index(st.session_state.ollama_model),
            key="model_select",
            help="Choose from your locally pulled Ollama models.",
        )
        st.session_state.ollama_model = selected
    else:
        st.markdown('<span class="status-offline">● Ollama Offline</span>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:0.82rem;color:#9d87b0;margin-top:0.6rem;line-height:1.6">
<b>To start the AI:</b><br>
1. Install <a href="https://ollama.com/download" target="_blank" style="color:#c084fc">ollama.com</a><br>
2. Run in a terminal:<br>
<code style="color:#c084fc">ollama pull llama3</code><br>
<code style="color:#c084fc">ollama serve</code><br>
3. Refresh this page 💜
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Session stats
    st.markdown(s["session_header"])
    n_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    col1, col2 = st.columns(2)
    col1.metric(s["messages_lbl"], n_msgs)
    col2.metric(s["duration_lbl"], get_session_duration())

    # ── Mood timeline
    if st.session_state.emotion_log:
        emo_data = st.session_state.emotion_log
        emotions = [e[1] for e in emo_data]
        colors   = [EMOTION_COLORS.get(e, "#c4b5fd") for e in emotions]
        EMO_EMOJI = {
            "joy": "🙂 joy", "sadness": "😔 sadness", "anger": "😠 anger",
            "fear": "😟 fear", "shame": "😳 shame", "neutral": "😐 neutral",
        }
        y_labels = [EMO_EMOJI.get(e, e) for e in emotions]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(emotions))), y=y_labels,
            mode="markers+lines",
            marker=dict(size=11, color=colors, line=dict(width=1.5, color="#7c3aed")),
            line=dict(color="#7c3aed", width=1.5, dash="dot"),
            hovertemplate="%{y}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=4, b=0), height=150,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(
                showgrid=False, tickfont=dict(color="#a78bba", size=10),
                categoryorder="array",
                categoryarray=[EMO_EMOJI[k] for k in EMO_GROUPS],
            ),
            font=dict(family="Inter"),
        )
        st.markdown(f"**{s['mood_lbl'].strip('*')}**")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        all_trigs = []
        for m in st.session_state.messages:
            all_trigs.extend(m.get("triggers", []))
        if all_trigs:
            top = Counter(all_trigs).most_common(8)
            tags_html = " ".join(f'<span class="tag">{t}</span>' for t, _ in top)
            st.markdown(f"**{s['themes_lbl'].strip('*')}**")
            st.markdown(tags_html, unsafe_allow_html=True)

    st.divider()

    # ── Privacy controls
    st.markdown(s["privacy_header"])
    if st.button(s["export_btn"], use_container_width=True):
        export = [{"role": m["role"], "content": m["content"], "ts": m["ts"]}
                  for m in st.session_state.messages]
        buf = io.BytesIO(json.dumps(export, ensure_ascii=False, indent=2).encode())
        st.download_button(
            s["download_btn"], buf,
            file_name="soul-ink-session.json",
            mime="application/json", use_container_width=True,
        )

    if st.button(s["delete_btn"], use_container_width=True):
        for k in ["messages", "emotion_log", "show_breathe", "show_ground"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else False
        st.session_state.session_start = time.time()
        st.rerun()

    st.markdown(f"""
<div style="font-size:0.72rem;color:#6b5a7a;margin-top:1rem;line-height:1.5">
{s['footer']}
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── About / Architecture panel ──────────────────────────────────────────
    with st.expander(s["arch_title"], expanded=False):
        for step in s["arch_steps"]:
            st.markdown(
                f"""<div class="pipeline-step">
  <div class="pipeline-num">{step['n']}</div>
  <div>
    <div class="pipeline-title">{step['title']}</div>
    <div class="pipeline-desc">{step['desc']}</div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='font-size:0.72rem;color:#6b5a7a;margin-top:0.7rem'>{s['arch_footer']}</div>",
            unsafe_allow_html=True,
        )

# ── Main chat area ────────────────────────────────────────────────────────────
_rtl_class = "rtl-mode" if st.session_state.language in ("ar",) else ""
st.markdown(f'<div class="{_rtl_class}">', unsafe_allow_html=True)

st.markdown(
    f"""
<div class="hero">
  <div class="hero-badge">💜</div>
  <div class="hero-text">
    {s["main_title"].lstrip('# ').strip()}
    <div class="hero-sub">{s["main_sub"]}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Welcome message ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="💜"):
        st.markdown(s["welcome"])

# ── Render message history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "💜" if msg["role"] == "assistant" else "🫂"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("show_crisis"):
            _cr = build_helpline_card(
                st.session_state.get("user_country", "Not specified"),
                st.session_state.get("language", "en"),
            )
            st.markdown(f'<div class="crisis-banner">{_cr}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)



# ── Breathing exercise widget ─────────────────────────────────────────────────
if st.session_state.show_breathe:
    st.markdown("""
<div style="text-align:center">
  <div class="breathe-circle">Breathe<br>in · hold<br>out · hold</div>
  <p style="color:#a78bba;font-size:0.85rem">4 counts in → 4 hold → 4 out → 4 hold<br>Repeat for 2 minutes 💜</p>
</div>
""", unsafe_allow_html=True)

# ── Grounding widget ──────────────────────────────────────────────────────────
if st.session_state.show_ground:
    with st.expander("🌿 5-4-3-2-1 Grounding Exercise", expanded=True):
        st.markdown("""
Take a slow breath, then notice:

**👁 5 things you can SEE** — look around and name them  
**✋ 4 things you can TOUCH** — feel their texture  
**👂 3 things you can HEAR** — listen carefully  
**👃 2 things you can SMELL** — or imagine your favourite scent  
**👅 1 thing you can TASTE** — or notice the sensation in your mouth  

This brings you back to the present moment. You're safe right here. 💜
        """)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(s["placeholder"])


if user_input:
    user_input = user_input.strip()
    if not user_input:
        st.stop()

    ts_now = datetime.utcnow().strftime("%H:%M")
    lower  = user_input.lower()

    # ── /lang override command
    _lang_cmd = re.match(r'^/lang\s+(en|dar|ar|fr)$', lower.strip())
    if _lang_cmd:
        _forced = _lang_cmd.group(1)
        st.session_state.language = _forced
        _names = {"en": "🇬🇧 English", "dar": "🇲🇦 Darija", "ar": "🌐 العربية", "fr": "🇫🇷 Français"}
        st.session_state.messages.append({"role": "user",      "content": user_input, "ts": ts_now, "triggers": []})
        st.session_state.messages.append({"role": "assistant", "content": f"Language set to {_names[_forced]}. I'll respond in that language from now on. 💜", "ts": ts_now, "triggers": []})
        st.rerun()

    # ── Auto-detect language from input ────────────────────────────────────
    # Priority: Arabic script > Franco-Darija > French > keep current
    if detect_darija_arabic(user_input):
        # Pure Arabic script (no ASCII letters) → Arabic; mixed → Darija
        _detected = "ar" if all(not c.isascii() or c.isspace() or not c.isalpha() for c in user_input) else "dar"
    elif detect_darija_franco(user_input):
        _detected = "dar"
    elif detect_french(user_input):
        _detected = "fr"
    else:
        _detected = "en"
    if _detected != st.session_state.language:
        st.session_state.language = _detected
        s = S()   # refresh strings after language change

    # ── Special commands ───────────────────────────────────────────────────
    if lower in {"/breathe", "breathe", "/breath"}:
        st.session_state.show_breathe = True
        st.session_state.show_ground  = False
        bot_reply = s["breathe_reply"]
        st.session_state.messages.append({"role": "user",      "content": user_input, "ts": ts_now, "triggers": []})
        st.session_state.messages.append({"role": "assistant", "content": bot_reply,  "ts": ts_now, "triggers": []})
        st.rerun()

    if lower in {"/ground", "ground", "grounding"}:
        st.session_state.show_ground  = True
        st.session_state.show_breathe = False
        bot_reply = s["ground_reply"]
        st.session_state.messages.append({"role": "user",      "content": user_input, "ts": ts_now, "triggers": []})
        st.session_state.messages.append({"role": "assistant", "content": bot_reply,  "ts": ts_now, "triggers": []})
        st.rerun()


    # ── Emotion & crisis detection
    is_crisis = detect_crisis(user_input)
    primary_emo, coarse_emo = detect_emotion(user_input)
    triggers = extract_triggers(user_input)

    if primary_emo:
        st.session_state.emotion_log.append((ts_now, primary_emo))

    st.session_state.messages.append({
        "role":    "user",
        "content": user_input,
        "ts":      ts_now,
        "emotion": primary_emo,
        "triggers": triggers,
    })

    with st.chat_message("user", avatar="🫂"):
        st.markdown(user_input)

    emo_ctx     = build_emotion_context(primary_emo, coarse_emo)
    _country    = st.session_state.get("user_country", "Not specified")
    _helpline   = HELPLINES_DB.get(_country, HELPLINES_DB["Not specified"])
    country_ctx = (
        f"\n\n[User country: {_country} | Emergency: {_helpline['emergency']} | "
        f"Crisis line: {_helpline['crisis']} ({_helpline['name']}) — "
        f"always mention this number if the user is in crisis or the app seems to be failing.]"
        if _country != "Not specified" else ""
    )
    system = s["system_prompt"].format(emotion_context=emo_ctx, country_context=country_ctx)

    show_crisis_in_reply = False

    with st.chat_message("assistant", avatar="💜"):
        online = check_ollama_status()

        if not online:
            # ── Ollama offline: warm rule-based fallback
            reply = s["fallback"].get(primary_emo, s["fallback"]["neutral"])
            if is_crisis:
                reply = s["crisis_fall"]
                show_crisis_in_reply = True
            st.warning(s["offline_warn"])
            st.markdown(reply)

        else:
            try:
                if is_crisis:
                    st.markdown(s["crisis_live"])
                    _cr_live = build_helpline_card(
                        st.session_state.get("user_country", "Not specified"),
                        st.session_state.get("language", "en"),
                    )
                    st.markdown(f'<div class="crisis-banner">{_cr_live}</div>', unsafe_allow_html=True)
                    show_crisis_in_reply = False

                with st.spinner(""):
                    reply = call_ollama(
                        st.session_state.ollama_model,
                        st.session_state.messages,
                        system,
                    )
                st.markdown(reply)

            except requests.exceptions.ConnectionError:
                reply = "I couldn't reach the local AI — please make sure Ollama is running (`ollama serve`). 💜"
                st.error("❌ Could not connect to Ollama. Run `ollama serve` in a terminal.")
            except requests.exceptions.Timeout:
                reply = "The model took too long to respond. It might still be loading — please try again in a moment. 💜"
                st.warning("⏱ Ollama response timed out.")
            except Exception as e:
                err = str(e)
                if "model" in err.lower() and ("not found" in err.lower() or "pull" in err.lower()):
                    reply = f"The model **{st.session_state.ollama_model}** isn't downloaded yet. Run `ollama pull {st.session_state.ollama_model}` in a terminal. 💜"
                    st.error(f"❌ Model not found: run `ollama pull {st.session_state.ollama_model}`")
                else:
                    reply = "Something went wrong. Please try again. 💜"
                    st.error(f"Error: {err[:200]}")

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    reply,
        "ts":         ts_now,
        "triggers":   [],
        "show_crisis": show_crisis_in_reply,
    })
