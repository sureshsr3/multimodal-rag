"""
Image processor — handles two very different image types:

SCENARIO 1 — Visual images (park, faces, objects, scenes):
  OCR finds no text → BLIP captioning generates a natural language description
  → caption is embedded with TextEncoder
  → user can ask "what's in the park photo?" and get a real answer

SCENARIO 2 — Text images (receipts, screenshots, scanned docs):
  OCR extracts text successfully → text is embedded with TextEncoder
  → user can ask "what was the total on the receipt?" and get a real answer

Both paths produce a text chunk in the same 512-d space.
No heavy ViT model needed — everything goes through TextEncoder.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


@dataclass
class ImageChunk:
    image: Image.Image
    source: str
    ocr_text: str = ""
    caption: str = ""
    modality: str = "image"
    metadata: dict = field(default_factory=dict)


@dataclass
class OCRTextChunk:
    text: str
    source: str
    chunk_type: str = "ocr"        # "ocr" | "caption"
    modality: str = "text"
    metadata: dict = field(default_factory=dict)


class ImageProcessor:
    # If OCR returns fewer than this many characters we consider the image
    # to be visual-only and run BLIP captioning instead.
    OCR_MIN_CHARS = 20

    def __init__(self, languages: list[str] | None = None):
        self.languages = languages or ["en"]
        self._blip_processor = None
        self._blip_model = None

    # ------------------------------------------------------------------ #
    #  OCR — for text images (receipts, screenshots, scanned docs)        #
    # ------------------------------------------------------------------ #

    def _ocr(self, image: Image.Image) -> str:
        """Try pytesseract first (fast), fall back to easyocr."""
        # pytesseract
        try:
            import pytesseract
            import shutil
            tess = shutil.which("tesseract")
            if tess:
                pytesseract.pytesseract.tesseract_cmd = tess
            text = pytesseract.image_to_string(image, lang="eng").strip()
            if text:
                return text
        except Exception:
            pass

        # easyocr fallback
        try:
            import easyocr
            import numpy as np
            reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
            results = reader.readtext(np.array(image), detail=0, paragraph=True)
            return "\n".join(results)
        except Exception:
            pass

        return ""

    # ------------------------------------------------------------------ #
    #  BLIP captioning — for visual images (parks, objects, scenes)       #
    # ------------------------------------------------------------------ #

    def _load_blip(self):
        """Lazy-load BLIP-large. Downloads ~900 MB on first call, cached forever after."""
        if self._blip_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            # large model generates significantly more detailed captions than base
            model_id = "Salesforce/blip-image-captioning-large"
            print(f"[ImageProcessor] Loading BLIP-large captioning model — first time takes a few minutes…")
            self._blip_processor = BlipProcessor.from_pretrained(model_id)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(model_id)
            self._blip_model.eval()
            print("[ImageProcessor] BLIP-large ready.")
        return self._blip_processor, self._blip_model

    def _caption(self, image: Image.Image) -> str:
        """
        Generate a rich, detailed description by running BLIP-large with
        multiple guided prompts and combining the results.
        """
        import torch
        processor, model = self._load_blip()

        # Guided prompts force BLIP to look at specific aspects of the scene
        prompts = [
            "a photo of",                          # free-form overall scene
            "the objects in this photo include",   # enumerate items
            "the setting of this photo is",        # location / environment
            "the colors and lighting in this photo are",  # visual detail
        ]

        parts = []
        for prompt in prompts:
            inputs = processor(image, prompt, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    num_beams=5,           # beam search → more coherent output
                    length_penalty=1.2,    # encourages longer descriptions
                )
            text = processor.decode(out[0], skip_special_tokens=True).strip()
            # Remove the prompt prefix if BLIP echoed it back
            if text.lower().startswith(prompt.lower()):
                text = text[len(prompt):].strip(" ,.")
            if text:
                parts.append(text)

        # Combine into one rich paragraph
        combined = ". ".join(p.rstrip(".") for p in parts if p) + "."
        return combined

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def process(self, image_path: str | Path) -> tuple[ImageChunk, OCRTextChunk | None]:
        """
        Returns:
            image_chunk  — always returned, holds the PIL image + metadata
            text_chunk   — the embeddable text: OCR text OR BLIP caption
                           None only if both OCR and captioning fail
        """
        path = str(image_path)
        img  = Image.open(path).convert("RGB")
        name = Path(path).name

        # ── Try OCR first (fast, no model download) ──────────────────
        ocr_text = self._ocr(img)
        caption  = ""

        if len(ocr_text) >= self.OCR_MIN_CHARS:
            # SCENARIO 2 — text image (receipt, screenshot, document)
            # Store ONLY the extracted text — no filename that could mislead the LLM
            embed_text  = ocr_text
            chunk_type  = "ocr"
            description = f"OCR ({len(ocr_text)} chars)"
        else:
            # SCENARIO 1 — visual image (park, photo, scene)
            # Store ONLY the BLIP caption — no filename that could mislead the LLM
            caption     = self._caption(img)
            embed_text  = caption
            chunk_type  = "caption"
            description = f"BLIP caption: '{caption}'"

        print(f"[ImageProcessor] {name} → {description}")

        image_chunk = ImageChunk(
            image=img,
            source=path,
            ocr_text=ocr_text,
            caption=caption,
            metadata={
                "ocr_text":   ocr_text,
                "caption":    caption,
                "chunk_type": chunk_type,
                "width":      img.width,
                "height":     img.height,
            },
        )

        text_chunk = OCRTextChunk(
            text=embed_text,
            source=path,
            chunk_type=chunk_type,
            metadata={
                "source_image": path,
                "chunk_type":   chunk_type,
                "ocr_text":     ocr_text,
                "caption":      caption,
            },
        ) if embed_text.strip() else None

        return image_chunk, text_chunk

    def process_batch(
        self, paths: list[str | Path]
    ) -> list[tuple[ImageChunk, OCRTextChunk | None]]:
        return [self.process(p) for p in paths]
