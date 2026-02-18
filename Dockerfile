# Slim image for x86 (e.g. Core 2 Duo). No GPU; OCR via Gemini API.
# Build this image ON the old PC so NumPy is compiled for its CPU (avoids X86_V2 error).
FROM python:3.11-slim-bookworm

WORKDIR /app

# Poppler, OpenCV deps, and build deps for NumPy (so we can build from source for old CPUs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    gcc \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Build NumPy from source so it matches the host CPU (no X86_V2 requirement). Then install rest.
RUN pip install --no-cache-dir "numpy<2" --no-binary numpy && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless

COPY main.py .

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8567

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8567"]
