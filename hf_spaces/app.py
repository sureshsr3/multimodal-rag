"""
Multimodal RAG — Hugging Face Spaces deployment
================================================
Single Streamlit app — no FastAPI, no Ollama, no Inference API.
LLM (TinyLlama-1.1B) is loaded directly on the Space CPU using transformers.
Vector store is in-memory (resets when the Space restarts).

Optional Spaces secret:  HF_TOKEN  (speeds up model downloads, not required)
Optional:                HF_LLM_MODEL  (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
"""
from __future__ import annotations
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Iterator

import torch
import streamlit as st

# ── Make src/ importable ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoders import MultimodalEncoder
from src.ingestion import PDFProcessor, ImageProcessor, AudioProcessor, TextProcessor
from src.vectorstore import MultimodalVectorStore
from src.rag import MultimodalRetriever
from src.rag.retriever import RetrievedChunk

# ── Config ───────────────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "")          # optional — speeds up downloads
LLM_MODEL   = os.environ.get("HF_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TOP_K       = int(os.environ.get("TOP_K", "5"))

SYSTEM_PROMPT = """Answer the user's question using ONLY the text provided below in the CONTEXT section.
Do not use any outside knowledge. Do not explain what you are doing. Do not repeat these instructions.
If the context does not contain the answer, say: "The uploaded files do not mention that."
End your answer with: Sources: <filename>"""

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    if "rag_ready" not in st.session_state:
        with st.spinner("Loading models (first launch takes ~2 min)…"):
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = MultimodalEncoder(output_dim=512)
            encoder.to(device).eval()

            store     = MultimodalVectorStore(persist_dir=None)   # in-memory
            retriever = MultimodalRetriever(encoder=encoder, store=store, top_k=TOP_K)

            st.session_state.update(
                rag_ready   = True,
                encoder     = encoder,
                store       = store,
                retriever   = retriever,
                pdf_proc    = PDFProcessor(),
                img_proc    = ImageProcessor(),
                audio_proc  = AudioProcessor(),
                text_proc   = TextProcessor(),
                device      = device,
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K
    if "cross_modal" not in st.session_state:
        st.session_state.cross_modal = True

_init_state()

# ── Helpers ────────────────────────────────────────────────────────────────────
def _detect_modality(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":                                        return "pdf"
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:  return "image"
    if suffix in {".mp3", ".mp4", ".wav", ".m4a", ".ogg"}:    return "audio"
    return "text"


def _ingest(file_bytes: bytes, filename: str) -> str:
    """Save to a temp file, ingest, return a status string."""
    s        = st.session_state
    store    = s["store"]
    encoder  = s["encoder"]
    modality = _detect_modality(filename)
    suffix   = Path(filename).suffix

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        store.delete_by_source(str(tmp_path))   # safe no-op on first upload
        ids_added: list = []

        if modality == "pdf":
            text_chunks, img_chunks = s["pdf_proc"].process(tmp_path)
            if text_chunks:
                texts = [c.text for c in text_chunks]
                embs  = encoder.encode_text(texts).detach().cpu().numpy()
                metas = [{"modality": "text", "source": filename, **c.metadata} for c in text_chunks]
                ids_added += store.add(embs, texts, metas)
            for c in (img_chunks or []):
                ocr = s["img_proc"]._ocr(c.image)
                if ocr.strip():
                    emb = encoder.encode_text([ocr]).detach().cpu().numpy()
                    ids_added += store.add(emb, [ocr], [{"modality": "image", "source": filename, **c.metadata}])

        elif modality == "image":
            img_chunk, ocr_chunk = s["img_proc"].process(tmp_path)
            if ocr_chunk:
                emb = encoder.encode_text([ocr_chunk.text]).detach().cpu().numpy()
                ids_added += store.add(emb, [ocr_chunk.text], [{"modality": "image", "source": filename, **img_chunk.metadata}])
            else:
                placeholder = f"[Image with no extractable text: {filename}]"
                ids_added += store.add(
                    encoder.encode_text([placeholder]).detach().cpu().numpy(),
                    [placeholder], [{"modality": "image", "source": filename}],
                )

        elif modality == "audio":
            chunks = s["audio_proc"].process(tmp_path)
            texts  = [c.text for c in chunks]
            embs   = encoder.encode_text(texts).detach().cpu().numpy()
            metas  = [{"modality": "audio", "source": filename, **c.metadata} for c in chunks]
            ids_added += store.add(embs, texts, metas)

        else:
            chunks = s["text_proc"].process_file(str(tmp_path))
            texts  = [c.text for c in chunks]
            embs   = encoder.encode_text(texts).detach().cpu().numpy()
            metas  = [{"modality": "text", "source": filename} for _ in chunks]
            ids_added += store.add(embs, texts, metas)

        return f"✓ {filename} → {len(ids_added)} chunks ({modality})"
    finally:
        tmp_path.unlink(missing_ok=True)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    sections = []
    for i, chunk in enumerate(chunks, 1):
        chunk_type  = chunk.metadata.get("chunk_type", "")
        modality    = chunk.modality
        source_name = Path(chunk.source).name

        if modality == "audio":
            label = "AUDIO RECORDING TRANSCRIPT"
        elif modality == "image" and chunk_type == "caption":
            label = "IMAGE DESCRIPTION"
        elif modality == "image":
            label = "TEXT FROM IMAGE"
        else:
            label = "DOCUMENT"

        sections.append(f"[{i}] {label} — {source_name}\n{chunk.document}")
    return "\n\n---\n\n".join(sections)


def _load_llm():
    """Load TinyLlama into session state (once per session)."""
    if "llm_pipe" not in st.session_state:
        import threading
        from transformers import pipeline, AutoTokenizer
        with st.spinner(f"Loading LLM `{LLM_MODEL}` — first launch ~2 min…"):
            tok  = AutoTokenizer.from_pretrained(LLM_MODEL, token=HF_TOKEN or None)
            pipe = pipeline(
                "text-generation",
                model=LLM_MODEL,
                tokenizer=tok,
                torch_dtype=torch.float32,   # CPU-safe
                device="cpu",
            )
        st.session_state.llm_pipe      = pipe
        st.session_state.llm_tokenizer = tok


def _stream_answer(query: str, chunks: list[RetrievedChunk], history: list[dict]) -> Iterator[str]:
    """
    Stream tokens from TinyLlama running locally on the Space CPU.
    No external API, no provider, no token required.
    Uses transformers TextIteratorStreamer for live token output.
    """
    import threading
    from transformers import TextIteratorStreamer

    _load_llm()
    pipe      = st.session_state.llm_pipe
    tokenizer = st.session_state.llm_tokenizer

    context_block = _build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    # Embed context directly in the user turn so small models focus on it
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {query}"
    })

    # Apply the model's built-in chat template (handles <|system|>, <|user|> etc.)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def _generate():
        pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            streamer=streamer,
        )

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()

    for token in streamer:
        yield token


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Multimodal RAG")
    st.caption("Voice memos · Receipts · PDFs · Images")
    st.info(
        "⚠️ **HF Spaces demo**: the knowledge base is **in-memory** "
        "and resets when the Space restarts. Re-upload your files after restart.",
        icon="ℹ️",
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    st.header("Upload Documents")
    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "png", "jpg", "jpeg", "mp3", "mp4", "wav", "m4a", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Ingest Files", type="primary"):
        for f in uploaded:
            with st.spinner(f"Processing **{f.name}**…"):
                try:
                    msg = _ingest(f.getvalue(), f.name)
                    st.success(msg)
                except Exception as exc:
                    st.error(f"✗ {f.name}: {exc}")

    # ── Search settings ───────────────────────────────────────────────────────
    st.divider()
    st.header("Search Settings")
    st.session_state.top_k = st.slider("Results per query", 1, 20, st.session_state.top_k)
    st.session_state.cross_modal = st.toggle(
        "Cross-modal search",
        value=st.session_state.cross_modal,
        help="Search all modalities separately and merge results",
    )

    # ── Indexed sources ───────────────────────────────────────────────────────
    st.divider()
    st.header("Indexed Files")
    store   = st.session_state["store"]
    sources = store.list_sources()

    if not sources:
        st.info("No files indexed yet.")
    else:
        st.caption(f"Total chunks: {store.count()}")
        for src in sources:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"📄 `{Path(src).name}`")
            if c2.button("🗑️", key=f"del_{src}", help="Remove from knowledge base"):
                store.delete_by_source(src)
                st.success(f"Deleted {Path(src).name}")
                st.rerun()

    st.divider()
    if st.button("🗑️ Clear ALL", type="secondary"):
        if st.checkbox("Confirm — wipe everything"):
            store.reset()
            st.session_state.messages = []
            st.success("Knowledge base cleared.")
            st.rerun()

    st.divider()
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Device: `{st.session_state.get('device', '?')}`")
    st.caption(f"LLM: `{LLM_MODEL}`")

# ── Main chat window ───────────────────────────────────────────────────────────
st.header("Chat with your documents")
st.caption("Ask questions across all uploaded files. Correct the model — it remembers the conversation.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask anything…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    retriever = st.session_state["retriever"]
    retriever.top_k = st.session_state.top_k

    # Retrieve
    if st.session_state.cross_modal:
        cross  = retriever.retrieve_cross_modal(user_input, top_k_per_modality=st.session_state.top_k)
        chunks = [c for lst in cross.values() for c in lst]
    else:
        chunks = retriever.retrieve(user_input, top_k=st.session_state.top_k)

    if not chunks:
        with st.chat_message("assistant"):
            msg = "I couldn't find any relevant documents. Please upload some files first."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        st.stop()

    # Show retrieved sources
    with st.expander(f"📚 Retrieved {len(chunks)} chunk(s)", expanded=False):
        for chunk in chunks:
            badge = {"audio": "🎙️", "image": "🖼️", "text": "📄"}.get(chunk.modality, "❓")
            st.markdown(
                f"**{badge} {chunk.modality.upper()}** · score `{chunk.score:.3f}` · "
                f"`{Path(chunk.source).name}`\n\n> {chunk.document[:300]}…"
            )

    # Stream answer
    history = st.session_state.messages[:-1]
    with st.chat_message("assistant"):
        answer_box  = st.empty()
        full_answer = ""
        for token in _stream_answer(user_input, chunks, history):
            full_answer += token
            answer_box.markdown(full_answer + "▌")
        answer_box.markdown(full_answer)

    if full_answer:
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
