
# SOUL-INK — The AI That Listens (Anonymous, No DB)
# 6dainn neon-purple aesthetic • Voice input (upload) • Streamlit

import io, json, time, tempfile, os
from collections import Counter
from datetime import datetime

import streamlit as st
import plotly.express as px

# ---------- Privacy: session-only ----------
if "entries" not in st.session_state: st.session_state.entries = []
if "start_ts" not in st.session_state: st.session_state.start_ts = time.time()
if "voice_text" not in st.session_state: st.session_state.voice_text = ""

# ---------- Lazy model loaders ----------
@st.cache_resource
def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_emotion_pipe():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
    model = "joeddav/distilbert-base-uncased-go-emotions"
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True, function_to_apply="sigmoid")
    return pipe, mdl.config.id2label

nlp = load_spacy()
emo_pipe, id2label = load_emotion_pipe()

# ---------- Config ----------
EMO_GROUPS = {
    "joy": {"joy","amusement","excitement","gratitude","love","optimism","pride","relief","admiration"},
    "sadness": {"sadness","disappointment","embarrassment","grief","remorse"},
    "anger": {"anger","annoyance","disgust"},
    "fear": {"fear","nervousness","apprehension"},
    "shame": {"guilt","shame"},
    "neutral": {"neutral"}
}

QUESTION_SETS = {
    "joy": [
        "What exactly made this feel good—and how can you recreate it tomorrow?",
        "If you bottled this moment into one habit, what would it be?"
    ],
    "sadness": [
        "What loss or unmet need sits underneath this feeling?",
        "If a kind friend spoke now, what would they say word-for-word?"
    ],
    "anger": [
        "Which boundary was crossed? What outcome would feel fair and specific?",
        "Is anger protecting a softer emotion (hurt, fear)? What signals that?"
    ],
    "fear": [
        "What is the feared outcome, and how likely is it (0–100%) realistically?",
        "What would ‘prepared’ look like in the next 10 minutes?"
    ],
    "shame": [
        "Whose standards are you using to judge yourself right now?",
        "What concrete evidence challenges the harsh self-story?"
    ],
    "neutral": [
        "What matters most today, and why that—not something else?",
        "Name one tiny action that moves you 1% toward it."
    ]
}

COPING = {
    "joy": [
        "Savoring: write 3 lines about what went right and how to repeat it.",
        "Gratitude micro-note: message someone who contributed to this moment."
    ],
    "sadness": [
        "Reframe: list 2 facts vs 2 interpretations; challenge the interpretation.",
        "Activation: pick one 10-minute task that slightly improves your day."
    ],
    "anger": [
        "Boundary script: I feel __ when __. I need __. If not, I will __.",
        "Box breathing 4-4-4-4 for 2 minutes; then write the request you’ll make."
    ],
    "fear": [
        "Grounding 5-4-3-2-1 (see, touch, hear, smell, taste).",
        "Control check: list 3 worries; tag each controllable/not; act on controllable."
    ],
    "shame": [
        "Self-compassion: write to yourself exactly as you would to a close friend.",
        "Evidence scan: list proof for/against the core negative belief."
    ],
    "neutral": [
        "Body scan: relax jaw/shoulders; 2 minutes of breath.",
        "Tiny win: choose a 5-minute action that’s meaningful, not perfect."
    ]
}

CRISIS_KEYWORDS = {
    "suicide","kill myself","end it","self harm","self-harm","cutting",
    "no reason to live","overdose","i want to die","i want die","i want end"
}

DISCLAIMER = (
    "SOUL-INK is not therapy or medical advice. If you feel unsafe or in crisis, "
    "contact local emergency services or a trusted hotline."
)

# ---------- Helpers ----------
def detect_crisis(text:str)->bool:
    t = text.lower()
    return any(k in t for k in CRISIS_KEYWORDS)

def coarse_emotions(raw_scores, threshold=0.30, top_k=3):
    grouped = {g:0.0 for g in EMO_GROUPS}
    for s in raw_scores:
        lab, sc = s["label"], float(s["score"])
        for g, members in EMO_GROUPS.items():
            if lab in members and sc > grouped[g]:
                grouped[g] = sc
    keep = {k:round(v,3) for k,v in grouped.items() if v>=threshold}
    if keep: return keep
    return {k:round(v,3) for k,v in dict(sorted(grouped.items(), key=lambda x:x[1], reverse=True)[:top_k]).items()}

def extract_triggers(text:str, max_terms=6):
    doc = nlp(text)
    chunks = [c.text.strip().lower() for c in doc.noun_chunks if len(c.text.strip())>1]
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
    common = [w for w,_ in Counter(tokens).most_common(10)]
    seen, out = set(), []
    for x in chunks + common:
        if x not in seen:
            seen.add(x); out.append(x)
        if len(out)>=max_terms: break
    return out

# ---------- Theme ----------
st.set_page_config(page_title="SOUL-INK — The AI That Listens", page_icon="💜", layout="wide")
st.markdown('''
<style>
:root { --ink: #a855f7; }
.stApp { background: radial-gradient(1200px 800px at 20% 10%, #1b0f28 0%, #0b0712 60%, #08060d 100%); }
h1, h2, h3, h4 { color: #e9d5ff !important; }
.block-container { padding-top: 1.5rem; }
.stButton>button { background: var(--ink); color:white; border-radius:12px; }
</style>
''', unsafe_allow_html=True)

# ---------- Header ----------
st.title("SOUL-INK — The AI That Listens")
st.caption("6dainn neon-purple • Anonymous by design (no database) • Voice or text")
st.warning(DISCLAIMER)

with st.expander("Privacy & Controls", expanded=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("Export session (.json)"):
            buf = io.BytesIO(json.dumps(st.session_state.entries, ensure_ascii=False, indent=2).encode("utf-8"))
            st.download_button("Download export", buf, file_name=f"soul-ink_{int(time.time())}.json", mime="application/json")
    with c2:
        if st.button("Delete all data"):
            st.session_state.entries = []
            st.success("Session data deleted.")
    with c3:
        st.write(f"Session length: {int(time.time()-st.session_state.start_ts)}s")

left, right = st.columns([0.55, 0.45])

with left:
    st.subheader("Tell me what’s on your mind")

    # ---------- VOICE INPUT (upload -> transcription) ----------
    st.write("Prefer to talk instead of type?")
    audio_file = st.file_uploader("Upload a short voice note (MP3/WAV/M4A)", type=["mp3","wav","m4a"])
    if audio_file is not None:
        # Save uploaded audio to temp
        tmp_in = tempfile.NamedTemporaryFile(delete=False)
        tmp_in.write(audio_file.read()); tmp_in.flush(); tmp_in.close()

        # Convert to WAV if needed using pydub/ffmpeg
        in_path = tmp_in.name
        ext = (audio_file.name.split(".")[-1] or "").lower()
        wav_path = in_path
        if ext != "wav":
            try:
                from pydub import AudioSegment
                sound = AudioSegment.from_file(in_path, format=ext)
                wav_path = in_path + ".wav"
                sound.export(wav_path, format="wav")
            except Exception as e:
                st.error(f"Audio conversion failed: {e}")

        # Transcribe with SpeechRecognition (Google free recognizer)
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            with sr.AudioFile(wav_path) as src:
                audio_data = r.record(src)
            text_from_audio = r.recognize_google(audio_data)
            st.session_state.voice_text = text_from_audio
            st.success("Voice transcribed:")
            st.write(text_from_audio)
        except Exception as e:
            st.error(f"Could not transcribe audio: {e}")

    # ---------- Unified text area (prefills with transcribed voice) ----------
    user_input = st.text_area(
        "Type or edit your thought below:",
        value=st.session_state.get("voice_text", ""),
        height=170,
        placeholder="Speak or write whatever’s on your mind..."
    )

    if st.button("Reflect", type="primary"):
        txt = (user_input or "").strip()
        if not txt:
            st.info("Say or write a few sentences first.")
        else:
            if detect_crisis(txt):
                st.error("I’m hearing crisis language. You matter. Please consider reaching out to a trusted person or local emergency services.")
            preds = emo_pipe(txt)[0]
            top_raw = sorted(preds, key=lambda d: d["score"], reverse=True)[:6]
            coarse = coarse_emotions(top_raw, threshold=0.30, top_k=3)
            primary = max(coarse, key=lambda k: coarse[k]) if coarse else "neutral"
            trigs = extract_triggers(txt)

            qs = QUESTION_SETS.get(primary, QUESTION_SETS["neutral"])[:2]
            acts = COPING.get(primary, COPING["neutral"])[:2]

            row = {
                "ts": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                "text": txt, "emotions": coarse, "primary": primary,
                "triggers": trigs, "questions": qs, "actions": acts
            }
            st.session_state.entries.append(row)

            st.success("I’m here. Here’s what I’m noticing and wondering:")
            st.markdown(f"**Primary emotion:** `{primary}`  •  **Signals:** " +
                        (", ".join([f"{k}:{v}" for k,v in coarse.items()]) if coarse else "—"))
            if trigs:
                st.markdown("**Possible themes:** " + ", ".join(trigs))
            st.markdown("**Questions to explore**")
            for q in qs: st.write("• " + q)
            st.markdown("**Coping steps**")
            for a in acts: st.write("• " + a)

with right:
    st.subheader("Your week at a glance")
    if st.session_state.entries:
        times = [e["ts"] for e in st.session_state.entries]
        prims = [e["primary"] for e in st.session_state.entries]
        df = {"time": times, "emotion": prims}
        fig = px.scatter(df, x="time", y=["emotion"]*len(times), hover_name=prims, title="Primary emotion timeline")
        fig.update_traces(marker_size=12, selector=dict(mode="markers"))
        fig.update_yaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

        all_trigs = []
        for e in st.session_state.entries: all_trigs.extend(e["triggers"])
        if all_trigs:
            from collections import Counter
            cts = Counter(all_trigs)
            st.markdown("**Top themes:** " + ", ".join([f"{k} ({v})" for k,v in cts.most_common(20)]))
    else:
        st.info("Insights appear after your first reflection.")

st.caption("Built for NavHacks 2025 • Transformers + spaCy + Streamlit • Voice upload • Anonymous by design.")
