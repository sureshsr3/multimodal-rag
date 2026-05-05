# Multimodal RAG System

> Ask questions across voice memos, scanned receipts, PDFs, and images — all at once, running entirely on your machine for free.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20phi3-black)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What This Is

A production-ready **Retrieval-Augmented Generation (RAG)** system that understands **four data modalities simultaneously**:

| Modality | File Types | How It Works |
|---|---|---|
| **Text** | `.txt` | Sentence-boundary chunking → MiniLM embeddings |
| **PDF** | `.pdf` | pdfplumber text + PyMuPDF image extraction → OCR/embed |
| **Image** | `.jpg`, `.png` | OCR path (receipts) or BLIP-large caption path (photos) |
| **Audio/Video** | `.mp3`, `.mp4`, `.wav`, `.m4a` | Whisper ASR → transcript → MiniLM embeddings |

**Example query:** *"What was the total spend mentioned in the voice memo vs the scanned receipts?"*
The system retrieves audio transcripts, image OCR text, and PDF content in a single search and generates a grounded answer.

**100% free and local** — no OpenAI, no Anthropic, no cloud APIs. Everything runs on your machine.

---

## Architecture

```
Input Files
    │
    ├── PDF  ──────► pdfplumber + PyMuPDF ──────────────────────┐
    ├── Image ─────► pytesseract / BLIP-large captioning ────────┤
    ├── Audio ─────► Whisper (base) → transcript ────────────────┤
    └── Text  ─────► sentence chunker ──────────────────────────┤
                                                                  │
                                             MultimodalEncoder    │
                              all-MiniLM-L6-v2 + MLP head (512-d)│
                                                                  ▼
                                                     ChromaDB (cosine / HNSW)
                                                                  │
User Query ──► QueryEncoder (same 512-d space) ──► top-k chunks  │
                                                                  ▼
                                             phi3 via Ollama (local LLM)
                                                                  │
                                                             Answer ◄──────── Streamlit Chat UI
```

### Component Map

| Component | Technology | Why This Choice |
|---|---|---|
| Text backbone | `all-MiniLM-L6-v2` | 90 MB, 5× faster than BERT, trained on 1B sentence pairs |
| Image (visual) | BLIP-large (Salesforce) | Generates rich multi-sentence captions, no ViT training needed |
| Image (text) | pytesseract + EasyOCR | Fast, no model download for structured text (receipts, docs) |
| Audio | OpenAI Whisper base | 74M params, multilingual, word timestamps |
| Vector store | ChromaDB | Embedded, persistent, metadata filtering, cosine HNSW |
| LLM | phi3 via Ollama | 3.8B params, instruction-tuned, 100% local, free |
| API | FastAPI + uvicorn | Async, streaming responses, background ingestion |
| Frontend | Streamlit | Rapid UI with full chat history |

---

## Key Design Decisions

### 1. Single 512-d vector space for all modalities
All modalities are projected to the same 512-dimensional L2-normalized space using a custom PyTorch `MultimodalEncoder` with per-modality MLP projection heads. This means **one query searches audio, image, and text chunks simultaneously**.

### 2. Two-path image processing
```
Image uploaded
    │
    ├── OCR finds ≥ 20 chars?  ──YES──► Use OCR text  (receipts, docs, screenshots)
    │
    └──────────────────────────NO───► Run BLIP-large captioning  (photos, scenes)
                                       4 guided prompts · beam_search · num_beams=5
```

### 3. Non-blocking ingestion
Heavy operations (Whisper, BLIP) can take 2–10 minutes. Upload returns a `job_id` immediately and the frontend polls for completion — the API never blocks.

### 4. Anti-hallucination layers
- Filenames are **never stored** inside document text (prevents LLM using filename keywords)
- System prompt strictly forbids using training knowledge
- `temperature=0.2` for deterministic, factual answers
- phi3 chosen over tinyllama — far lower hallucination rate

### 5. Full chat history
Every query sends the complete conversation to the LLM so you can say *"that's wrong, look more carefully"* and it self-corrects.

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/multimodal-rag.git
cd multimodal-rag

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull the LLM
```bash
ollama pull phi3
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 4. Start the API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 5. Start the frontend (new terminal)
```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501** and start uploading files!

---

## Docker (one-command start)

> Requires Docker Desktop + Ollama running on the host machine.

```bash
docker compose -f docker/docker-compose.yml up --build
```

- API available at `http://localhost:8000`
- UI available at `http://localhost:8501`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a file, returns `job_id` immediately |
| `GET` | `/job/{job_id}` | Poll ingestion status (`pending` / `done` / `error`) |
| `POST` | `/query` | Retrieval only — returns top-k chunks (<1s) |
| `POST` | `/query/stream` | Streaming LLM answer (tokens sent live) |
| `GET` | `/sources` | List all indexed files |
| `GET` | `/stats` | Chunk counts by modality |
| `DELETE` | `/source` | Remove a specific file from the knowledge base |
| `POST` | `/reset` | Wipe everything |
| `GET` | `/health` | Health check + device info |

**Example query:**
```bash
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the total expense?", "top_k": 5, "cross_modal": true}'
```

---

## Project Structure

```
multimodal-rag/
├── src/
│   ├── encoders/
│   │   ├── text_encoder.py          # MiniLM + MLP projection head → 512-d
│   │   ├── image_encoder.py         # ViT backbone (optional, BLIP path preferred)
│   │   ├── audio_encoder.py         # Whisper → transcript → TextEncoder
│   │   └── multimodal_encoder.py   # Unified encoder + NT-Xent contrastive loss
│   ├── ingestion/
│   │   ├── text_processor.py        # Sentence-boundary chunking (512 tok, 64 overlap)
│   │   ├── pdf_processor.py         # pdfplumber text + PyMuPDF image extraction
│   │   ├── image_processor.py       # OCR path / BLIP-large caption path
│   │   └── audio_processor.py       # imageio-ffmpeg → numpy → Whisper → chunks
│   ├── vectorstore/
│   │   └── store.py                 # ChromaDB wrapper (cosine HNSW, metadata filter)
│   ├── rag/
│   │   ├── retriever.py             # Dense retrieval + cross-modal retrieval
│   │   └── generator.py            # Ollama streaming generator + chat history
│   └── api/
│       └── main.py                 # FastAPI app (non-blocking upload, streaming query)
├── frontend/
│   └── app.py                      # Streamlit chat UI
├── training/
│   ├── dataset.py                  # Contrastive pair dataset
│   └── train.py                    # NT-Xent fine-tuning loop
├── tests/
│   ├── test_encoders.py
│   └── test_ingestion.py
├── docker/
│   ├── Dockerfile                  # API container
│   ├── Dockerfile.frontend         # Streamlit container
│   └── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## How Cross-Modal Search Works

When `cross_modal=True`, the retriever queries ChromaDB **separately for each modality** and merges results:

```python
results = {
    "text":  retrieve(query, filter={"modality": "text"},  top_k=5),
    "image": retrieve(query, filter={"modality": "image"}, top_k=5),
    "audio": retrieve(query, filter={"modality": "audio"}, top_k=5),
}
```

This guarantees representation from every modality in the LLM context — not just whichever modality happened to score highest.

---

## Contrastive Training (Optional Fine-tuning)

The encoder supports NT-Xent contrastive loss (from SimCLR/CLIP) to further align modalities:

```python
loss = contrastive_loss(audio_embeddings, text_embeddings, temperature=0.07)
```

- **Temperature = 0.07**: standard from SimCLR/CLIP papers — sharp similarity distribution
- **Symmetric**: computed A→B and B→A, averaged
- **No hard negative mining needed**: full batch negatives used automatically

```bash
python training/train.py --epochs 10 --batch-size 32 --lr 1e-4
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `phi3` | Ollama model to use for generation |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `UPLOAD_DIR` | `./data/uploads` | Where uploaded files are saved |
| `CHROMA_DIR` | `./data/chroma_db` | ChromaDB persistence directory |
| `ENCODER_CHECKPOINT` | *(empty)* | Path to fine-tuned encoder `.pt` file |
| `TOP_K` | `5` | Default number of chunks to retrieve |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

- **PyTorch 2.2** — custom encoder, projection heads, NT-Xent loss
- **Transformers 4.40** — MiniLM, BLIP-large
- **OpenAI Whisper** — audio transcription
- **ChromaDB 0.5** — vector store with HNSW index
- **FastAPI 0.111** — async REST API with streaming
- **Streamlit 1.35** — chat frontend
- **Ollama** — local LLM serving (phi3, llama3, tinyllama)
- **imageio-ffmpeg** — bundled ffmpeg (no system install needed)
- **pytesseract + EasyOCR** — OCR for text images

---

## License

MIT — use freely, attribution appreciated.
