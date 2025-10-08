# Lecture Voice → Notes Generator

A Streamlit app that turns recorded lectures into transcripts, summaries, knowledge graphs, and flashcards using Whisper, Hugging Face summarization models, and spaCy.

## Features
- Transcribe audio (mp3, wav, m4a) with OpenAI Whisper
- Chunked summarization using a Hugging Face pipeline
- Concept extraction and interactive knowledge graph (pyvis + networkx)
- Automatic flashcard generation from lecture text
- Simple local processing; works with CPU or GPU

## Requirements
- Python 3.8+
- ffmpeg on PATH
- pip packages (examples): streamlit, openai-whisper, transformers, torch, spacy, networkx, pyvis

## Installation
1. Clone the repository:
   git clone <repo-url>
2. Create and activate a virtual environment:
   python -m venv .venv
   .venv\Scripts\activate  (Windows) or source .venv/bin/activate (Unix)
3. Install dependencies:
   pip install -r requirements.txt
4. Ensure ffmpeg is installed and available on PATH.

## SpaCy model
The app attempts to load `en_core_web_sm` and will prompt to download it if missing:
python -m spacy download en_core_web_sm

## Running
Start the Streamlit app:
streamlit run app.py

Upload a lecture audio file and use the UI buttons to generate:
- Transcript (Whisper)
- Summary (Hugging Face summarizer)
- Knowledge Graph (spaCy + networkx + pyvis)
- Flashcards (spaCy-based Q/A)

## Notes & Tips
- For faster transcription and summarization, use a machine with a CUDA-capable GPU and install compatible torch builds.
- Large lectures are chunked for summarization to avoid model input limits.
- Audio files are stored temporarily and removed after transcription.

## Troubleshooting
- "ffmpeg not found": install ffmpeg and add to PATH.
- SpaCy model download fails: run the download command manually in the environment.
- Summarizer model loads slowly or times out on CPU—consider using smaller model sizes or GPU.

## License
MIT