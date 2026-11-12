# Cognivault-Production-Grade-Agentic-RAG-LLM-Infrastructure
Cognivault is a production-ready AI orchestration platform built for reliable LLM applications, featuring data ingestion and processing, an LLM Gateway with integrated guardrails, intelligent planning for retrieval and context-memory decisions, Agentic RAG, embedding and model failover, response caching, and automated evaluation pipelines.

## Run the whole application in Docker

Build one image from the project root:

```powershell
docker build -t cognivault .
```

Run the API, chat UI, and evaluation UI together. Keep `.env` private and provide it at runtime:

```powershell
docker run --env-file .env -p 8000:8000 -p 8501:8501 -p 8502:8502 cognivault
```

- Chat UI: http://localhost:8502
- Evaluation UI: http://localhost:8501
- API: http://localhost:8000

The container uses the configured cloud services and API credentials from `.env`. Local virtual environments, secrets, and `Data/` and `processed_data/` folders are excluded from the image build.
