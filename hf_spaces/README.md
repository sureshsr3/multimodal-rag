---
title: Multimodal RAG
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Multimodal RAG — HF Spaces Demo

Ask questions across voice memos, scanned receipts, PDFs, and images — all at once.

## Setup

Add your Hugging Face token as a **Space secret**:
- Key: `HF_TOKEN`
- Value: your token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)

Optionally set `HF_LLM_MODEL` (default: `mistralai/Mistral-7B-Instruct-v0.3`).

## Note

This is a demo deployment. The vector store is **in-memory** and resets when the Space restarts.
For persistent storage, run the full stack locally with Docker.

See the [full project on GitHub](https://github.com/YOUR_USERNAME/multimodal-rag).
