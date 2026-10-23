FROM python:3.11.16-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/services/inference:/app/packages HOME=/tmp \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
COPY requirements/inference.txt /tmp/inference.txt
RUN pip install --no-cache-dir --disable-pip-version-check \
    --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/inference.txt \
    && pip check && rm /tmp/inference.txt
COPY services/inference/vf_inference /app/services/inference/vf_inference
COPY packages/vf_ml/__init__.py packages/vf_ml/transforms.py /app/packages/vf_ml/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=2s --timeout=3s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "vf_inference.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
