# Slim image for x86 (e.g. Core 2 Duo). No GPU; OCR via Gemini API.
FROM python:3.11-slim-bookworm

WORKDIR /app

# Poppler for pdf2image; headless OpenCV deps (no GUI/GL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Use headless OpenCV in container to avoid heavy GUI libs
RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless

COPY main.py .

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8567

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8567"]
