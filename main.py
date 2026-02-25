import asyncio
import io
import itertools
import json
import os
import threading
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

def _load_gemini_api_keys() -> List[str]:
    keys = []
    # Check for comma-separated keys
    keys_str = os.getenv("GEMINI_API_KEYS")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not keys:
        # Check for GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
        for i in range(1, 10):
            key = os.getenv(f"GEMINI_API_KEY_{i}") or os.getenv(f"GEMINI_API_KEY{i}")
            if key:
                keys.append(key.strip())
    if not keys:
        # Fallback to single key
        single_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if single_key:
            keys.append(single_key.strip())
    return keys

GEMINI_API_KEYS = _load_gemini_api_keys()


class ApiKeyPool:
    def __init__(self, keys: List[str]):
        self.all_keys = keys
        self.active_keys = set(keys)
        self.current_idx = 0
        self.lock = threading.Lock()

    def get_key(self) -> str:
        with self.lock:
            if not self.active_keys:
                raise HTTPException(
                    status_code=429,
                    detail="All Gemini API keys have exceeded their quota. Please try again later.",
                )
            # Find the next active key
            for _ in range(len(self.all_keys)):
                key = self.all_keys[self.current_idx]
                self.current_idx = (self.current_idx + 1) % len(self.all_keys)
                if key in self.active_keys:
                    return key
            raise HTTPException(status_code=429, detail="No active keys available.")

    def disable_key(self, key: str):
        with self.lock:
            if key in self.active_keys:
                self.active_keys.remove(key)
                print(f"API key ending in ...{key[-4:]} disabled due to 429 error. {len(self.active_keys)} keys remaining.")


_api_key_pool = ApiKeyPool(GEMINI_API_KEYS)


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


class LineItem(BaseModel):
    company_from: str = ""
    company_to: str = ""
    invoice_number: str = ""
    goods_products: str = ""
    quantity: str = ""
    unit_price: str = ""
    whole_price_amount: str = ""
    total_price: str = ""


class OcrResponse(BaseModel):
    pages: List[Page]
    metadata: OcrMetadata
    markdown: str = ""
    line_items: List[LineItem] = Field(default_factory=list)


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


def _get_next_api_key() -> str:
    if not _api_key_pool.all_keys:
        raise HTTPException(
            status_code=500,
            detail="No Gemini API keys are set in the environment. Please set GEMINI_API_KEYS, or GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.",
        )
    return _api_key_pool.get_key()


def _parse_gemini_json(output_text: str) -> dict:
    """Parse Gemini output text into a JSON dict, with repair fallback."""
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

    return parsed


def _extract_response_text(response) -> str:
    """
    Extract the actual text response from a Gemini response,
    skipping any 'thought' parts (returned when thinking is enabled).
    """
    if not response.candidates or not response.candidates[0].content.parts:
        return ""
    for part in response.candidates[0].content.parts:
        # Skip thinking parts (they have a 'thought' attribute set to True)
        if getattr(part, "thought", False):
            continue
        if part.text:
            return part.text
    return ""


LINE_ITEM_FIELDS = ["company_from", "company_to", "invoice_number", "goods_products", "quantity", "unit_price", "whole_price_amount", "total_price"]


def _normalize_line_items(raw_items: list) -> List[dict]:
    """Sanitize line items: strip $ from prices, ensure all fields present."""
    result = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        cleaned = {}
        for field in LINE_ITEM_FIELDS:
            val = str(item.get(field, "")).strip()
            # Strip $ and commas from price fields
            if field in ("unit_price", "whole_price_amount", "total_price") and val:
                # Remove currency prefixes like $, US$, USD
                for prefix in ("US$", "USD", "$"):
                    if val.upper().startswith(prefix.upper()):
                        val = val[len(prefix):].strip()
                        break
                try:
                    float(val.replace(",", ""))
                    val = val.replace(",", "")
                except (ValueError, AttributeError):
                    pass  # Keep non-numeric values like "FOC" as-is
            cleaned[field] = val
        result.append(cleaned)
    return result


def run_gemini_ocr_on_image(pil_img: Image.Image, page_number: int) -> dict:
    """
    Run Gemini 3 Flash on a single image in two passes:
      Pass 1: Extract layout elements + markdown (what the model is good at).
      Pass 2: Extract CSV using the markdown as grounding context (prevents hallucination).
    """
    api_key = _get_next_api_key()

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    # ── Pass 1: Layout + Markdown extraction ──────────────────────────────
    layout_prompt = """Please output the layout information from the image as a single JSON object. You MUST include the structured elements AND a complete markdown representation in the same JSON output.

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

6. Final Output: Output ONLY a single JSON object with:
   - "elements": array where each element has "bbox", "category", and "text" (omit "text" for Picture).
   - "markdown": a string containing the complete document content as markdown, in English. This must be the full readable document in markdown format with proper structure (headers, lists, tables as markdown tables). Do not omit the markdown field.
"""

    # Retry loop for API keys
    max_retries = len(_api_key_pool.all_keys) if _api_key_pool.all_keys else 1
    
    # ── Pass 1: Layout + Markdown extraction ──────────────────────────────
    for attempt in range(max_retries):
        api_key = _get_next_api_key()
        try:
            client = genai.Client(api_key=api_key)
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            layout_response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[image_part, layout_prompt],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0
                    )
                ),
            )
            break  # Success
        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "quota" in err_msg.lower():
                _api_key_pool.disable_key(api_key)
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=429, detail="All Gemini API keys exhausted quota (layout pass).")
                continue # Try next key
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Gemini API error (layout pass): {exc}",
                ) from exc
    else:
        raise HTTPException(status_code=502, detail="Request failed after exhausting all API keys.")

    layout_text = _extract_response_text(layout_response)
    if not layout_text:
        raise HTTPException(status_code=502, detail="Gemini returned empty response (layout pass).")
    parsed = _parse_gemini_json(layout_text)

    # Fix literal \n escape sequences in markdown
    if "markdown" in parsed and isinstance(parsed["markdown"], str):
        parsed["markdown"] = parsed["markdown"].replace("\\n", "\n")

    markdown_content = parsed.get("markdown", "")

    # ── Pass 2: Line-item extraction grounded on the markdown ──────────────
    items_prompt = f"""Extract invoice line-item data from this document as a JSON array.

Use the REFERENCE TEXT below as your primary source. Cross-reference with the image to verify.

Each line item must be a JSON object with exactly these fields:
- "company_from": the seller/shipper/consignor company name (from the document header). Use a short name.
- "company_to": the buyer/consignee company name (from the document header). Use a short name.
- "invoice_number": the invoice number from the document header
- "goods_products": description of the goods/product for this line item
- "quantity": the quantity
- "unit_price": the unit price for this line item (number only, no currency symbol)
- "whole_price_amount": the line-item total / amount (number only, no currency symbol). If "FOC" or "Free", keep as-is.
- "total_price": the invoice grand total (same value on every row, number only, no currency symbol)

Rules:
- One object per line item found in the invoice. Do NOT include total/summary rows.
- company_from, company_to, invoice_number, and total_price are document-level fields — repeat them on every item.
- Copy numbers exactly as they appear in the document. Do not recalculate or reformat.
- If a field does not exist in the document, set it to an empty string "".
- If a value is "FOC", "Free", "N/A" etc., keep it as-is.
- Translate non-English item descriptions to English.
- Do NOT invent or guess values that are not in the document.

REFERENCE TEXT:
---
{markdown_content}
---

Output ONLY a JSON object with a single key "line_items" containing the array of line item objects.
You MUST include at least one item if the document contains any line items.
"""

    for attempt in range(max_retries):
        api_key = _get_next_api_key()
        try:
            client = genai.Client(api_key=api_key)
            items_response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[image_part, items_prompt],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=1024
                    )
                ),
            )
            break  # Success
        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "quota" in err_msg.lower():
                _api_key_pool.disable_key(api_key)
                if attempt == max_retries - 1:
                    parsed["line_items"] = []
                    return parsed
                continue # Try next key
            else:
                parsed["line_items"] = []
                return parsed
    else:
        parsed["line_items"] = []
        return parsed

    items_text = _extract_response_text(items_response)
    if items_text:
        items_parsed = _parse_gemini_json(items_text)
        raw_items = items_parsed.get("line_items", [])
        if isinstance(raw_items, list):
            parsed["line_items"] = _normalize_line_items(raw_items)
        else:
            parsed["line_items"] = []
    else:
        parsed["line_items"] = []

    return parsed


async def process_invoice_file(file: UploadFile, max_pages: int) -> OcrResponse:
    if max_pages <= 0:
        raise HTTPException(status_code=400, detail="max_pages must be positive.")

    content_type = (file.content_type or "").lower()
    filename = file.filename or "unknown_file"

    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file '{filename}': {exc}") from exc

    if not contents:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{filename}' is empty.")

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
        raise HTTPException(status_code=400, detail=f"Failed to open image '{filename}': {exc}") from exc
    images = [img]

    # Run Gemini 3 Flash on each page and convert output to our schema
    pages: List[Page] = []
    markdown_parts: List[str] = []
    all_line_items: List[LineItem] = []
    for idx, img in enumerate(images, start=1):
        ocr_output = await asyncio.to_thread(run_gemini_ocr_on_image, img, page_number=idx)
        page = convert_elements_to_page(ocr_output, page_number=idx, img_width=img.width, img_height=img.height)
        pages.append(page)
        page_markdown = ocr_output.get("markdown", "")
        if page_markdown:
            markdown_parts.append(page_markdown)
        page_items = ocr_output.get("line_items", [])
        for item_dict in page_items:
            try:
                all_line_items.append(LineItem(**item_dict))
            except Exception:
                pass  # Skip malformed items

    metadata = OcrMetadata(
        source_type=source_type,
        model=GEMINI_MODEL_NAME,
        processed_pages=len(pages),
    )

    return OcrResponse(
        pages=pages,
        metadata=metadata,
        markdown="\n\n".join(markdown_parts),
        line_items=all_line_items,
    )


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
    return await process_invoice_file(file, max_pages)


@app.post(
    "/ocr/bulk",
    response_model=List[OcrResponse],
    tags=["OCR"],
    summary="Run OCR on multiple invoice images in parallel",
    description=(
        "Upload multiple invoice images. "
        "Images are sent to Gemini 3 Flash for OCR in parallel. "
        "Returns a list of detailed JSON layouts including text blocks and tables."
    ),
)
async def ocr_invoice_bulk(
    files: List[UploadFile] = File(..., description="List of invoice images."),
    max_pages: int = 5,
) -> List[OcrResponse]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    tasks = [process_invoice_file(file, max_pages) for file in files]
    responses = await asyncio.gather(*tasks)
    return list(responses)


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

