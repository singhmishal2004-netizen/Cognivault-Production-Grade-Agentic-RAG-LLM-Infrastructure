FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BACKEND_URL=http://127.0.0.1:8000 \
    COGNIVAULT_API_URL=http://127.0.0.1:8000/query

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY app/ ./app/
COPY UI/ ./UI/
COPY evals/ ./evals/
COPY docker_entrypoint.py .

EXPOSE 8000 8501 8502

CMD ["python", "docker_entrypoint.py"]
