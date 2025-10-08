import streamlit as st
import tempfile
import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from collections import Counter

# ML libraries
import whisper
from transformers import pipeline
import torch
import networkx as nx
from pyvis.network import Network

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Lecture Voice-to-Notes Generator", layout="wide")
st.title("🎧 Lecture Voice → Notes Generator")

# ----------------- UTILITIES -----------------
def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        st.warning("ffmpeg not found on PATH. Whisper needs ffmpeg to process many audio formats. "
                   "Install ffmpeg or ensure it's on PATH for best results.")
        return False
    return True

def ensure_spacy_model():
    import importlib
    try:
        importlib.import_module("spacy")
    except Exception:
        st.error("spaCy is not installed. Please install spaCy in the environment.")
        return None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.info("Downloading spaCy 'en_core_web_sm' model (this runs once)...")
        try:
            subprocess.check_call([os.sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception as e:
            st.error(f"Failed to download/load spaCy model: {e}")
            return None

# ----------------- MODEL LOADING -----------------
@st.cache_resource
def load_whisper_model(size="tiny"):
    return whisper.load_model(size)

@st.cache_resource
def load_summarizer(model_name="sshleifer/distilbart-cnn-12-6"):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model=model_name, device=device)

# Load light models once (spacy handled lazily below)
with st.spinner("Preparing environment..."):
    ffmpeg_ok = ensure_ffmpeg()
    try:
        whisper_model = load_whisper_model("tiny")
    except Exception as e:
        st.error(f"Could not load Whisper model: {e}")
        whisper_model = None
    try:
        summarizer = load_summarizer()
    except Exception as e:
        st.error(f"Could not load summarization model: {e}")
        summarizer = None

# ----------------- HELPER FUNCTIONS -----------------
def save_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

def chunk_text(text, max_chars=900):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if not current:
            current = s
        elif len(current) + len(s) + 1 <= max_chars:
            current += " " + s
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks

def extract_concepts(text, top_k=15, nlp=None):
    if nlp is None:
        return []
    doc = nlp(text)
    candidates = [ent.text.strip() for ent in doc.ents] + [nc.text.strip() for nc in doc.noun_chunks]
    freq = Counter([c.lower() for c in candidates if len(c.split()) <= 4])
    return [k for k, _ in freq.most_common(top_k)]

def build_knowledge_graph(text, nodes):
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        present = [n for n in nodes if n.lower() in s.lower()]
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                a, b = present[i], present[j]
                if G.has_edge(a, b):
                    G[a][b]['weight'] = G[a][b].get('weight', 0) + 1
                else:
                    G.add_edge(a, b, weight=1)
    return G

def render_graph_html(G):
    net = Network(height="450px", width="100%", notebook=False)
    net.from_nx(G)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)
    try:
        return Path(tmp.name).read_text()
    finally:
        tmp.close()

def generate_flashcards(text, n=5, nlp=None):
    if nlp is None:
        return []
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
    cards = []
    for s in sents:
        if len(cards) >= n:
            break
        sent_nlp = nlp(s)
        ents = [ent.text for ent in sent_nlp.ents]
        if not ents:
            # fallback: pick a noun chunk as answer
            nc = [nc.text for nc in sent_nlp.noun_chunks]
            if not nc:
                continue
            answer = nc[0]
        else:
            answer = ents[0]
        question = s.replace(answer, "______")
        if question == s:
            # if replace didn't change (case mismatch), do case-insensitive replace
            question = re.sub(re.escape(answer), "______", s, flags=re.IGNORECASE)
        cards.append({"question": question, "answer": answer})
    return cards

# ----------------- UI -----------------
uploaded_file = st.file_uploader("Upload a lecture audio file (.mp3, .wav, .m4a)", type=["mp3","wav","m4a"])

if uploaded_file:
    st.info("Processing audio...")
    path = save_uploaded_file(uploaded_file)

    # Transcription
    if whisper_model is None:
        st.error("Whisper model not available. Cannot transcribe.")
    else:
        with st.spinner("Transcribing audio with Whisper..."):
            try:
                transcription = whisper_model.transcribe(path).get("text", "").strip()
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                transcription = ""
        try:
            os.remove(path)
        except Exception:
            pass

        st.subheader("📝 Transcript")
        st.text_area("Transcript", transcription, height=200)

        # Summary
        summary_btn = st.button("Generate Summary")
        if summary_btn:
            if not summarizer:
                st.error("Summarizer model not loaded.")
            else:
                max_length = st.slider("Summary Max Length", 50, 500, 150)
                min_length = st.slider("Summary Min Length", 20, 200, 40)
                with st.spinner("Generating summary..."):
                    chunks = chunk_text(transcription)
                    # Summarize each chunk and join
                    summaries = []
                    for c in chunks:
                        try:
                            out = summarizer(c, max_length=max_length, min_length=min_length, do_sample=False)
                            if isinstance(out, list) and len(out) > 0 and "summary_text" in out[0]:
                                summaries.append(out[0]["summary_text"])
                            elif isinstance(out, dict) and "summary_text" in out:
                                summaries.append(out["summary_text"])
                        except Exception as e:
                            st.warning(f"Chunk summarization failed: {e}")
                    final_summary = " ".join(summaries).strip()
                st.subheader("📌 Summary")
                st.write(final_summary or "No summary produced.")
                st.download_button("Download Summary", final_summary, file_name="summary.txt")

        # Knowledge Graph
        kg_btn = st.button("Build Knowledge Graph")
        if kg_btn:
            nlp = ensure_spacy_model()
            if not nlp:
                st.error("spaCy model not available; cannot build knowledge graph.")
            else:
                top_k = st.slider("Number of Concepts", 5, 30, 15)
                with st.spinner("Building knowledge graph..."):
                    nodes = extract_concepts(transcription, top_k, nlp=nlp)
                    if not nodes:
                        st.warning("No concepts extracted.")
                    G = build_knowledge_graph(transcription, nodes)
                    html = render_graph_html(G)
                st.subheader("📚 Knowledge Graph")
                st.components.v1.html(html, height=500, scrolling=True)

        # Flashcards
        fc_btn = st.button("Generate Flashcards")
        if fc_btn:
            nlp = ensure_spacy_model()
            if not nlp:
                st.error("spaCy model not available; cannot generate flashcards.")
            else:
                card_count = st.slider("Number of Flashcards", 1, 20, 5)
                with st.spinner("Creating flashcards..."):
                    cards = generate_flashcards(transcription, n=card_count, nlp=nlp)
                st.subheader("🎯 Flashcards")
                if not cards:
                    st.write("No flashcards generated.")
                for i, c in enumerate(cards, 1):
                    st.markdown(f"**Q{i}.** {c['question']}")
                    with st.expander("Answer"):
                        st.write(c['answer'])
                st.download_button("Download Flashcards", json.dumps(cards, indent=2), file_name="flashcards.json")


