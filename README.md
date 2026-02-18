# Invoice OCR API (FastAPI + Qwen3 VL 30B A3B Thinking)

This is a minimal FastAPI service that performs OCR on invoice images or PDFs by:

- Converting PDF pages to images
- Preprocessing each page (grayscale + contrast enhancement) with OpenCV
- Sending the images to the `qwen/qwen3-vl-30b-a3b-thinking` model via the OpenRouter API
- Returning a detailed JSON layout of the document (pages, text blocks, tables, etc.)

## Setup

From the project root:

```bash
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

The app uses `python-dotenv` and loads `.env` with `override=True`, so values in `.env` take precedence over OS environment variables.

## Running the server

```bash
uvicorn main:app --reload
```

Then open Swagger UI at:

- `http://127.0.0.1:8000/docs`

Use the `/ocr` endpoint to upload an image or PDF and inspect the JSON OCR result.

