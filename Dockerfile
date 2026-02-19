# Slim image for OCR via Gemini API.
FROM python:3.11-slim-bookworm

WORKDIR /app

# Poppler for pdf2image (PDF support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8567

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8567"]
