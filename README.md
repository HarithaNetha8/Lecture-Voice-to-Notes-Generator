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

## output

<img width="1357" height="719" alt="Screenshot 2025-10-08 174116" src="https://github.com/user-attachments/assets/b7355817-77e1-4190-84f1-9e756c338e10" />

<img width="1360" height="714" alt="Screenshot 2025-10-08 174321" src="https://github.com/user-attachments/assets/e28d12d5-9c3e-476b-b2cd-1d5be1c9bc0d" />

<img width="1316" height="513" alt="Screenshot 2025-10-08 174335" src="https://github.com/user-attachments/assets/776eaa3e-e77d-46e7-b21d-7fcfa6f45c9a" />

<img width="1347" height="552" alt="Screenshot 2025-10-08 174417" src="https://github.com/user-attachments/assets/da8fcbc9-7d77-415b-af38-e9d944e27ee1" />

<img width="1311" height="524" alt="Screenshot 2025-10-08 174457" src="https://github.com/user-attachments/assets/4ad8d1b9-70f9-478a-933c-9b38120b2920" />

<img width="1334" height="402" alt="Screenshot 2025-10-08 174522" src="https://github.com/user-attachments/assets/e14e8a65-7068-4132-aa9f-81790fcdd43c" />

<img width="1334" height="402" alt="Screenshot 2025-10-08 174522" src="https://github.com/user-attachments/assets/3c40d1fc-f357-438b-ab9e-78e206c3dd63" />

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
