import io
import json
import os
from typing import List, Literal, Optional, Union

from json_repair import repair_json

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field, ValidationError
from PIL import Image


# Load environment variables from .env (with .env taking precedence if both are set)
load_dotenv(override=True)

GEMINI_MODEL_NAME = "gemini-3-flash-preview"
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
    markdown: str = ""
    csv: str = ""


app = FastAPI(
    title="Invoice OCR API",
    description="OCR invoice images via Gemini 3 Flash (Google AI).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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


def _ensure_gemini_api_key() -> str:
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY or GOOGLE_API_KEY is not set in the environment.",
        )
    return GEMINI_API_KEY


def run_gemini_ocr_on_image(pil_img: Image.Image, page_number: int) -> dict:
    """
    Run Gemini 3 Flash on a single image and return JSON with elements (bbox, category, text).
    """
    _ensure_gemini_api_key()

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    prompt = """Please output the layout information from the image as a single JSON object. You MUST include the structured elements, a complete markdown representation, AND a CSV extraction of invoice fields in the same JSON output.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Translation Rules:
    - Keep all English text as-is.
    - Translate all Chinese and Arabic text to English.
    - Keep all numbers (digits 0-9) exactly as they appear in the original image, do not translate or convert them.
    - Maintain the original formatting and structure.

5. Constraints:
    - All layout elements must be sorted according to human reading order.

6. CSV Extraction:
    - Extract invoice-specific fields into a CSV string.
    - The CSV header row MUST be: invoice_number,date,unit_price,total_price,quantity,item,model,country_of_origin,country_of_export
    - If there are multiple line items, output one CSV row per item.
    - If a field is not found in the document, leave it empty (e.g. ,,).
    - All values must be in English. Translate any non-English values to English.

7. Final Output: Output ONLY a single JSON object. It MUST contain all three:
   - "elements": array where each element has "bbox", "category", and "text" (omit "text" for Picture).
   - "markdown": a string containing the complete document content as markdown, in English. This must be the full readable document in markdown format with proper structure (headers, lists, tables as markdown tables). Do not omit the markdown field.
   - "csv": a string containing the CSV data with the header row and one or more data rows. Do not omit the csv field.
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

    output_text = output_text.strip()

    # Strip any surrounding markdown code fences (```json ... ```)
    if output_text.startswith("```"):
        lines = output_text.splitlines()
        output_text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    # Extract the outermost JSON object
    start_idx = output_text.find("{")
    end_idx = output_text.rfind("}") + 1
    if start_idx >= 0 and end_idx > start_idx:
        candidate = output_text[start_idx:end_idx]
    else:
        candidate = output_text

    # Try strict parse first, fall back to json_repair
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        try:
            repaired = repair_json(candidate)
            parsed = json.loads(repaired)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini returned non-JSON output that could not be repaired: {exc}. Raw: {output_text[:500]}",
            ) from exc

    # Fix literal \n escape sequences in markdown and csv fields
    if "markdown" in parsed and isinstance(parsed["markdown"], str):
        parsed["markdown"] = parsed["markdown"].replace("\\n", "\n")
    if "csv" in parsed and isinstance(parsed["csv"], str):
        parsed["csv"] = parsed["csv"].replace("\\n", "\n")

    return parsed


@app.post(
    "/ocr",
    response_model=OcrResponse,
    tags=["OCR"],
    summary="Run OCR on an invoice image",
    description=(
        "Upload an invoice as an image. "
        "Images are sent to Gemini 3 Flash for OCR. "
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

    CSV_HEADER = "invoice_number,date,unit_price,total_price,quantity,item,model,country_of_origin,country_of_export"

    # Run Gemini 3 Flash on each page and convert output to our schema
    pages: List[Page] = []
    markdown_parts: List[str] = []
    csv_rows: List[str] = []
    for idx, img in enumerate(images, start=1):
        ocr_output = run_gemini_ocr_on_image(img, page_number=idx)
        page = convert_elements_to_page(ocr_output, page_number=idx, img_width=img.width, img_height=img.height)
        pages.append(page)
        page_markdown = ocr_output.get("markdown", "")
        if page_markdown:
            markdown_parts.append(page_markdown)
        page_csv = ocr_output.get("csv", "")
        if page_csv:
            for line in page_csv.strip().splitlines():
                if line.strip().lower() == CSV_HEADER.lower():
                    continue
                if line.strip():
                    csv_rows.append(line.strip())

    csv_output = CSV_HEADER + "\n" + "\n".join(csv_rows) if csv_rows else CSV_HEADER

    metadata = OcrMetadata(
        source_type=source_type,
        model=GEMINI_MODEL_NAME,
        processed_pages=len(pages),
    )

    return OcrResponse(
        pages=pages,
        metadata=metadata,
        markdown="\n\n".join(markdown_parts),
        csv=csv_output,
    )


def _parse_bbox(bbox_list: list, img_width: int, img_height: int) -> BoundingBox:
    if len(bbox_list) != 4:
        return BoundingBox(x_min=0, y_min=0, x_max=img_width, y_max=img_height)

    x1, y1, x2, y2 = [float(v) for v in bbox_list]

    # If all values are in [0, 1], treat as normalized and convert to pixels
    if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
        x1, y1 = x1 * img_width, y1 * img_height
        x2, y2 = x2 * img_width, y2 * img_height

    # Ensure correct ordering
    x_min = int(min(x1, x2))
    x_max = int(max(x1, x2))
    y_min = int(min(y1, y2))
    y_max = int(max(y1, y2))

    # Clamp to image bounds
    x_min = max(0, min(x_min, img_width))
    x_max = max(0, min(x_max, img_width))
    y_min = max(0, min(y_min, img_height))
    y_max = max(0, min(y_max, img_height))

    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def convert_elements_to_page(ocr_output: dict, page_number: int, img_width: int, img_height: int) -> Page:
    """
    Convert OCR output (elements array with bbox, category, text) to our Page schema.
    """
    elements = ocr_output.get("elements", [])
    blocks: List[Block] = []
    
    for idx, elem in enumerate(elements):
        bbox_list = elem.get("bbox", [0, 0, 0, 0])
        bbox = _parse_bbox(bbox_list, img_width, img_height)
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

