"""
FastAPI backend: file upload, ingestion, and query endpoints.
Upload is non-blocking — returns a job_id immediately and processes in background.
"""
from __future__ import annotations
import os
import shutil
import uuid
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

import torch
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.encoders import MultimodalEncoder
from src.ingestion import PDFProcessor, ImageProcessor, AudioProcessor, TextProcessor
from src.vectorstore import MultimodalVectorStore
from src.rag import MultimodalRetriever, RAGGenerator

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./data/uploads"))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", "./data/chroma_db"))
CHECKPOINT  = os.environ.get("ENCODER_CHECKPOINT", "")
TOP_K       = int(os.environ.get("TOP_K", "5"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
#  In-memory job tracker                                               #
# ------------------------------------------------------------------ #
# job_id -> {"status": "pending|done|error", "result": ..., "error": ...}
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)

# ------------------------------------------------------------------ #
#  App state                                                           #
# ------------------------------------------------------------------ #
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] Using device: {device}")

    print("[startup] Loading text encoder (all-MiniLM-L6-v2)…")
    encoder = MultimodalEncoder(output_dim=512)
    if CHECKPOINT and Path(CHECKPOINT).exists():
        encoder.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    encoder.to(device).eval()
    print("[startup] Encoder ready.")

    store     = MultimodalVectorStore(persist_dir=CHROMA_DIR)
    retriever = MultimodalRetriever(encoder=encoder, store=store, top_k=TOP_K)
    generator = RAGGenerator()

    _state.update(
        encoder=encoder,
        store=store,
        retriever=retriever,
        generator=generator,
        pdf_proc=PDFProcessor(),
        img_proc=ImageProcessor(),
        audio_proc=AudioProcessor(),
        text_proc=TextProcessor(),
        device=device,
    )
    print("[startup] All components ready. API is live.")
    yield
    _executor.shutdown(wait=False)
    _state.clear()


app = FastAPI(title="Multimodal RAG API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _detect_modality(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        return "image"
    if suffix in {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm"}:
        return "audio"
    return "text"


def _ingest_file(file_path: Path, filename: str) -> dict:
    s        = _state
    store    = s["store"]
    encoder  = s["encoder"]
    modality = _detect_modality(filename)
    ids_added: list = []

    if modality == "pdf":
        text_chunks, img_chunks = s["pdf_proc"].process(file_path)
        if text_chunks:
            texts = [c.text for c in text_chunks]
            embs  = encoder.encode_text(texts).detach().cpu().numpy()
            metas = [{"modality": "text", "source": str(file_path), **c.metadata} for c in text_chunks]
            ids_added += store.add(embs, texts, metas)
        if img_chunks:
            # embed via OCR text rather than ViT — fast and better for text-heavy PDFs
            for c in img_chunks:
                ocr = s["img_proc"]._ocr(c.image)
                if ocr.strip():
                    emb  = encoder.encode_text([ocr]).detach().cpu().numpy()
                    ids_added += store.add(emb, [ocr], [{"modality": "image", "source": str(file_path), **c.metadata}])

    elif modality == "image":
        img_chunk, ocr_chunk = s["img_proc"].process(file_path)
        if ocr_chunk:
            emb = encoder.encode_text([ocr_chunk.text]).detach().cpu().numpy()
            ids_added += store.add(
                emb, [ocr_chunk.text],
                [{"modality": "image", "source": str(file_path), **img_chunk.metadata}],
            )
        else:
            # no OCR text found — store a placeholder so the file is tracked
            ids_added += store.add(
                encoder.encode_text([f"Image: {filename}"]).detach().cpu().numpy(),
                [f"[Image with no extractable text: {filename}]"],
                [{"modality": "image", "source": str(file_path)}],
            )

    elif modality == "audio":
        chunks = s["audio_proc"].process(file_path)
        texts  = [c.text for c in chunks]
        embs   = encoder.encode_text(texts).detach().cpu().numpy()
        metas  = [{"modality": "audio", "source": str(file_path), **c.metadata} for c in chunks]
        ids_added += store.add(embs, texts, metas)

    else:  # plain text
        chunks = s["text_proc"].process_file(str(file_path))
        texts  = [c.text for c in chunks]
        embs   = encoder.encode_text(texts).detach().cpu().numpy()
        metas  = [{"modality": "text", "source": str(file_path)} for _ in chunks]
        ids_added += store.add(embs, texts, metas)

    return {"file": filename, "modality": modality, "chunks_indexed": len(ids_added)}


def _run_ingest_job(job_id: str, file_path: Path, filename: str):
    """Runs in a thread pool — does NOT block the API."""
    try:
        _jobs[job_id]["status"] = "processing"
        result = _ingest_file(file_path, filename)
        _jobs[job_id].update(status="done", result=result)
        print(f"[job {job_id}] done — {result}")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        _jobs[job_id].update(status="error", error=err)
        print(f"[job {job_id}] ERROR — {err}")


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Saves the file and starts ingestion in background. Returns job_id immediately.
    Always deletes old chunks for this filename before re-indexing so re-uploads are clean."""
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Auto-delete old chunks for this exact file so re-uploads don't stack up
    _state["store"].delete_by_source(str(dest))

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "filename": file.filename}
    _executor.submit(_run_ingest_job, job_id, dest, file.filename)

    return {"job_id": job_id, "filename": file.filename, "status": "processing"}


@app.get("/job/{job_id}")
async def job_status(job_id: str):
    """Poll this to check if ingestion finished."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


class ChatMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    modality_filter: str | None = None
    cross_modal: bool = False
    history: list[ChatMessage] = []   # previous chat turns


class ChunksResponse(BaseModel):
    chunks: list[dict]
    sources: list[str]


@app.post("/query", response_model=ChunksResponse)
async def query(req: QueryRequest):
    """
    Retrieval ONLY — fast (<1s). Returns matched chunks, no LLM call.
    The frontend calls /query/stream separately for the actual answer.
    """
    retriever = _state["retriever"]

    if req.cross_modal:
        cross  = retriever.retrieve_cross_modal(req.query, top_k_per_modality=req.top_k)
        chunks = [c for lst in cross.values() for c in lst]
    else:
        chunks = retriever.retrieve(req.query, top_k=req.top_k, modality_filter=req.modality_filter)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found. Upload some files first.")

    return ChunksResponse(
        chunks=[{"document": c.document[:300], "modality": c.modality,
                 "score": c.score, "source": c.source} for c in chunks],
        sources=list({c.source for c in chunks}),
    )


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streams tokens from the LLM as they are generated — no timeout issues."""
    retriever = _state["retriever"]
    generator = _state["generator"]

    if req.cross_modal:
        cross  = retriever.retrieve_cross_modal(req.query, top_k_per_modality=req.top_k)
        chunks = [c for lst in cross.values() for c in lst]
    else:
        chunks = retriever.retrieve(req.query, top_k=req.top_k, modality_filter=req.modality_filter)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found. Upload some files first.")

    history = [{"role": m.role, "content": m.content} for m in req.history]

    def token_stream():
        try:
            for token in generator.stream_tokens(req.query, chunks, history=history):
                yield token
        except RuntimeError as exc:
            yield f"\n\n[ERROR] {exc}"

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.get("/sources")
async def list_sources():
    return {"sources": _state["store"].list_sources()}


@app.get("/stats")
async def stats():
    store = _state["store"]
    return {
        "total": store.count(),
        "by_modality": {m: store.count(m) for m in ("text", "image", "audio")},
    }


@app.delete("/source")
async def delete_source(source: str):
    """Remove all chunks for one file from ChromaDB."""
    _state["store"].delete_by_source(source)
    # Also delete the file from uploads if it exists
    p = Path(source)
    if p.exists():
        p.unlink()
    return {"deleted": source}


@app.post("/reset")
async def reset_all():
    """Wipe the entire ChromaDB collection and all uploaded files."""
    _state["store"].reset()
    # Clear uploads folder too
    for f in UPLOAD_DIR.iterdir():
        if f.is_file():
            f.unlink()
    return {"status": "reset complete"}


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(_state.get("device", "unknown"))}
