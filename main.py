import io
import json
import os
from typing import List, Literal, Optional, Union

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai import types
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field, ValidationError
from PIL import Image
import cv2


# Load environment variables from .env (with .env taking precedence if both are set)
load_dotenv(override=True)

GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Word(BaseModel):
    text: str
    bbox: BoundingBox


class Line(BaseModel):
    text: str
    bbox: BoundingBox
    words: List[Word] = Field(default_factory=list)


class TextBlock(BaseModel):
    id: str
    type: Literal["text"]
    bbox: BoundingBox
    lines: List[Line] = Field(default_factory=list)


class TableCell(BaseModel):
    row: int
    column: int
    row_span: int = 1
    col_span: int = 1
    text: str
    bbox: BoundingBox


class TableBlock(BaseModel):
    id: str
    type: Literal["table"]
    bbox: BoundingBox
    rows: int
    columns: int
    cells: List[TableCell] = Field(default_factory=list)


Block = Union[TextBlock, TableBlock]


class Page(BaseModel):
    page_number: int
    width: Optional[int] = None
    height: Optional[int] = None
    blocks: List[Block] = Field(default_factory=list)


class OcrMetadata(BaseModel):
    source_type: Literal["image", "pdf"]
    model: str
    processed_pages: int


class OcrResponse(BaseModel):
    pages: List[Page]
    metadata: OcrMetadata


app = FastAPI(
    title="Invoice OCR API",
    description="OCR invoice images via Gemini 2.5 Flash (Google AI).",
    version="0.1.0",
)


@app.get("/health", tags=["Health"], summary="Health check")
async def health_check() -> dict:
    return {"status": "ok"}


def pdf_bytes_to_images(data: bytes, max_pages: int) -> List[Image.Image]:
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")

    # pdf2image requires Poppler; allow configuring its path via POPPLER_PATH env var.
    poppler_path = os.getenv("POPPLER_PATH")
    images = convert_from_bytes(
        data,
        first_page=1,
        last_page=max_pages,
        poppler_path=poppler_path,
    )
    return images


def preprocess_image(pil_img: Image.Image) -> Image.Image:
    rgb = pil_img.convert("RGB")
    np_img = np.array(rgb)

    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return Image.fromarray(enhanced, mode="L")


def _ensure_gemini_api_key() -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY or GOOGLE_API_KEY is not set in the environment.",
        )
    return GEMINI_API_KEY


def run_gemini_ocr_on_image(pil_img: Image.Image, page_number: int) -> dict:
    """
    Run Gemini 2.5 Flash on a single image and return JSON with elements (bbox, category, text).
    """
    _ensure_gemini_api_key()

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Translation Rules:
    - Translate all English and Chinese words to Arabic.
    - Keep all numbers (digits 0-9) exactly as they appear in the original image, do not translate or convert them.
    - Preserve any Arabic text as-is.
    - Maintain the original formatting and structure.

5. Constraints:
    - All layout elements must be sorted according to human reading order.

6. Final Output: The entire output must be a single JSON object with an "elements" array, where each element has "bbox", "category", and "text" (omit "text" for Picture).
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0  # Disable thinking for faster responses
                )
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {exc}",
        ) from exc

    if not response.candidates or not response.candidates[0].content.parts:
        raise HTTPException(status_code=502, detail="Gemini returned empty response.")

    output_text = response.candidates[0].content.parts[0].text or ""

    try:
        output_text = output_text.strip()
        start_idx = output_text.find("{")
        end_idx = output_text.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            return json.loads(output_text[start_idx:end_idx])
        raise ValueError("No JSON object found in output")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned non-JSON output: {exc}. Raw: {output_text[:500]}",
        )


@app.post(
    "/ocr",
    response_model=OcrResponse,
    tags=["OCR"],
    summary="Run OCR on an invoice image",
    description=(
        "Upload an invoice as an image. "
        "Images will be preprocessed (grayscale + contrast enhancement) "
        "and then sent to Gemini 2.5 Flash. "
        "Returns a detailed JSON layout including text blocks and tables."
    ),
)
async def ocr_invoice(
    file: UploadFile = File(..., description="Invoice image."),
    max_pages: int = 5,
) -> OcrResponse:
    if max_pages <= 0:
        raise HTTPException(status_code=400, detail="max_pages must be positive.")

    content_type = (file.content_type or "").lower()
    filename = file.filename or ""

    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}") from exc

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    is_image = content_type.startswith("image/") or any(
        filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    )

    if not is_image:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type or filename}. "
            "Please upload an image (PNG, JPG, TIFF, BMP). PDFs are not supported.",
        )

    images: List[Image.Image] = []
    source_type = "image"

    try:
        img = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {exc}") from exc
    images = [img]

    preprocessed_images = [preprocess_image(img) for img in images]

    # Run Gemini 2.5 Flash on each page and convert output to our schema
    pages: List[Page] = []
    for idx, img in enumerate(preprocessed_images, start=1):
        ocr_output = run_gemini_ocr_on_image(img, page_number=idx)
        page = convert_elements_to_page(ocr_output, page_number=idx, img_width=img.width, img_height=img.height)
        pages.append(page)

    metadata = OcrMetadata(
        source_type=source_type,
        model=GEMINI_MODEL_NAME,
        processed_pages=len(pages),
    )

    return OcrResponse(pages=pages, metadata=metadata)


def convert_elements_to_page(ocr_output: dict, page_number: int, img_width: int, img_height: int) -> Page:
    """
    Convert OCR output (elements array with bbox, category, text) to our Page schema.
    """
    elements = ocr_output.get("elements", [])
    blocks: List[Block] = []
    
    for idx, elem in enumerate(elements):
        bbox_list = elem.get("bbox", [0, 0, 0, 0])
        if len(bbox_list) != 4:
            bbox_list = [0, 0, 0, 0]
        bbox = BoundingBox(
            x_min=int(bbox_list[0]),
            y_min=int(bbox_list[1]),
            x_max=int(bbox_list[2]),
            y_max=int(bbox_list[3]),
        )
        category = elem.get("category", "Text")
        text_content = elem.get("text", "")
        
        if category == "Table":
            # Convert table HTML/text to TableBlock
            # For now, create a simple table block - you may want to parse HTML to extract cells
            table_block = TableBlock(
                id=f"table-{page_number}-{idx}",
                type="table",
                bbox=bbox,
                rows=1,  # Placeholder - would need to parse HTML
                columns=1,  # Placeholder
                cells=[TableCell(
                    row=0,
                    column=0,
                    text=text_content,
                    bbox=bbox,
                )],
            )
            blocks.append(table_block)
        else:
            # Text block (Text, Title, Formula, etc.)
            # Split text into lines (simple approach)
            lines_text = text_content.split("\n") if text_content else [""]
            lines: List[Line] = []
            for line_text in lines_text:
                if not line_text.strip():
                    continue
                line = Line(
                    text=line_text.strip(),
                    bbox=bbox,  # Use same bbox for line (simplified)
                    words=[],
                )
                lines.append(line)
            
            if not lines:
                # Ensure at least one line
                lines = [Line(text=text_content or "", bbox=bbox, words=[])]
            
            text_block = TextBlock(
                id=f"text-{page_number}-{idx}",
                type="text",
                bbox=bbox,
                lines=lines,
            )
            blocks.append(text_block)
    
    return Page(
        page_number=page_number,
        width=img_width,
        height=img_height,
        blocks=blocks,
    )

