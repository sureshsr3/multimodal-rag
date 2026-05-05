"""
PDF processor: extracts text per page and embedded images.
Uses pdfplumber for text (preserves layout) and fitz (PyMuPDF) for images.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import io

from PIL import Image


@dataclass
class PDFChunk:
    text: str
    page_number: int
    source: str
    modality: str = "text"
    metadata: dict = field(default_factory=dict)


@dataclass
class PDFImageChunk:
    image: Image.Image
    page_number: int
    image_index: int
    source: str
    modality: str = "image"
    metadata: dict = field(default_factory=dict)


class PDFProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = start + self.chunk_size
            chunks.append(" ".join(words[start:end]))
            start += self.chunk_size - self.chunk_overlap
        return [c for c in chunks if c.strip()]

    def extract_text_chunks(self, pdf_path: str | Path) -> list[PDFChunk]:
        import pdfplumber

        path = str(pdf_path)
        chunks = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                for chunk in self._chunk_text(text):
                    chunks.append(PDFChunk(
                        text=chunk,
                        page_number=page_num,
                        source=path,
                        metadata={"page": page_num, "total_pages": len(pdf.pages)},
                    ))
        return chunks

    def extract_images(self, pdf_path: str | Path) -> list[PDFImageChunk]:
        import fitz  # PyMuPDF

        path = str(pdf_path)
        doc = fitz.open(path)
        image_chunks = []

        for page_num, page in enumerate(doc, start=1):
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                pil_img = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
                image_chunks.append(PDFImageChunk(
                    image=pil_img,
                    page_number=page_num,
                    image_index=img_idx,
                    source=path,
                    metadata={"page": page_num, "image_index": img_idx},
                ))
        doc.close()
        return image_chunks

    def process(self, pdf_path: str | Path) -> tuple[list[PDFChunk], list[PDFImageChunk]]:
        return self.extract_text_chunks(pdf_path), self.extract_images(pdf_path)
