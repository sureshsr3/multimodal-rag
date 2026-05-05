"""
Streamlit chat UI for the Multimodal RAG system.
Supports full conversation history — correct the model and it re-checks.
"""
from __future__ import annotations
import os
import time
import requests
from pathlib import Path

import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🔍",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Session state                                                       #
# ------------------------------------------------------------------ #
if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role": "user"|"assistant", "content": "..."}

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "cross_modal" not in st.session_state:
    st.session_state.cross_modal = True

# ------------------------------------------------------------------ #
#  Sidebar                                                             #
# ------------------------------------------------------------------ #
with st.sidebar:
    st.title("🔍 Multimodal RAG")
    st.caption("Voice memos · Receipts · PDFs · Images")

    # ── Upload ────────────────────────────────────────────────────────
    st.header("Upload Documents")
    uploaded = st.file_uploader(
        "Drop files here",
        type=["pdf", "png", "jpg", "jpeg", "mp3", "mp4", "wav", "m4a", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Ingest Files", type="primary"):
        for f in uploaded:
            st.write(f"📤 Uploading **{f.name}**…")
            try:
                resp = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (f.name, f.getvalue(), f.type)},
                    timeout=30,
                )
                if not resp.ok:
                    st.error(f"Upload failed: {resp.text}")
                    continue

                job_id = resp.json()["job_id"]
                st.write(f"⚙️ Indexing `{f.name}`…")
                bar = st.progress(0)
                dots = 0
                while True:
                    time.sleep(3)
                    dots = (dots + 1) % 4
                    bar.progress(min(dots * 20, 80), text="Processing" + "." * (dots + 1))
                    job = requests.get(f"{API_URL}/job/{job_id}", timeout=10).json()
                    if job["status"] == "done":
                        bar.progress(100, text="Done!")
                        r = job["result"]
                        st.success(f"✓ {r['file']} → {r['chunks_indexed']} chunks ({r['modality']})")
                        break
                    elif job["status"] == "error":
                        bar.empty()
                        st.error(f"✗ {job['error']}")
                        break
            except requests.exceptions.Timeout:
                st.error("Upload timed out — is the API running?")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API on port 8000.")

    # ── Search settings ───────────────────────────────────────────────
    st.divider()
    st.header("Search Settings")
    st.session_state.top_k = st.slider("Results per query", 1, 20, st.session_state.top_k)
    st.session_state.cross_modal = st.toggle(
        "Cross-modal search",
        value=st.session_state.cross_modal,
        help="Search all modalities separately",
    )

    # ── Indexed sources with delete ───────────────────────────────────
    st.divider()
    st.header("Indexed Files")

    try:
        sources = requests.get(f"{API_URL}/sources", timeout=5).json()["sources"]
    except Exception:
        sources = []

    if not sources:
        st.info("No files indexed yet.")
    else:
        stats = {}
        try:
            stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        except Exception:
            pass

        st.caption(f"Total chunks: {stats.get('total', '?')}")
        for src in sources:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"📄 `{Path(src).name}`")
            if c2.button("🗑️", key=f"del_{src}", help="Delete from knowledge base"):
                try:
                    requests.delete(f"{API_URL}/source", params={"source": src}, timeout=10)
                    st.success(f"Deleted {Path(src).name}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    st.divider()
    if st.button("🗑️ Delete ALL", type="secondary"):
        if st.checkbox("Confirm — wipe everything"):
            try:
                requests.post(f"{API_URL}/reset", timeout=10)
                st.session_state.messages = []
                st.success("Knowledge base cleared.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # ── Clear chat ────────────────────────────────────────────────────
    st.divider()
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------ #
#  Main chat window                                                    #
# ------------------------------------------------------------------ #
st.header("Chat with your documents")
st.caption("You can correct the model — it remembers the conversation.")

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask anything… or say 'that's wrong, look more carefully'")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = {
        "query": user_input,
        "top_k": st.session_state.top_k,
        "modality_filter": None,
        "cross_modal": st.session_state.cross_modal,
        "history": st.session_state.messages[:-1],   # all turns except current
    }

    # Step 1: retrieval (fast)
    try:
        chunk_resp = requests.post(f"{API_URL}/query", json=payload, timeout=15)
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure uvicorn is running.")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("Retrieval timed out — API may still be starting up.")
        st.stop()

    if chunk_resp.status_code == 404:
        with st.chat_message("assistant"):
            msg = "I couldn't find any relevant documents. Please upload some files first."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        st.stop()
    elif not chunk_resp.ok:
        st.error(f"API error {chunk_resp.status_code}: {chunk_resp.text}")
        st.stop()

    chunks = chunk_resp.json()["chunks"]

    # Show retrieved sources in an expander
    with st.expander(f"📚 Retrieved {len(chunks)} chunk(s)", expanded=False):
        for chunk in chunks:
            badge = {"audio": "🎙️", "image": "🖼️", "text": "📄"}.get(chunk["modality"], "❓")
            st.markdown(
                f"**{badge} {chunk['modality'].upper()}** · score `{chunk['score']:.3f}` · "
                f"`{Path(chunk['source']).name}`\n\n> {chunk['document'][:300]}…"
            )

    # Step 2: stream answer
    with st.chat_message("assistant"):
        answer_box  = st.empty()
        full_answer = ""

        try:
            with requests.post(
                f"{API_URL}/query/stream",
                json=payload,
                stream=True,
                timeout=300,
            ) as stream_resp:
                if not stream_resp.ok:
                    st.error(f"Stream error: {stream_resp.text}")
                    st.stop()
                for raw in stream_resp.iter_content(chunk_size=None):
                    if raw:
                        full_answer += raw.decode("utf-8")
                        answer_box.markdown(full_answer + "▌")
            answer_box.markdown(full_answer)

        except requests.exceptions.Timeout:
            if full_answer:
                answer_box.markdown(full_answer)
                st.warning("Stream timed out — partial answer shown.")
            else:
                st.error("No response received. Check that Ollama is running.")

    # Save assistant reply to history
    if full_answer:
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
