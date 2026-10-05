# Doc-lens

RAG document assistant with a **FastAPI backend** and two frontends:

- **Next.js UI** in `app/`
- **Streamlit UI** in `frontend/`

## Project Structure

```
Doc-lens/
├── app/                    ← Next.js UI (client-side document assistant)
├── backend/
│   ├── api.py               ← FastAPI app (entry point)
│   ├── dependencies.py      ← Singleton DI (pipeline, store)
│   ├── schemas.py           ← All Pydantic request/response models
│   └── routers/
│       ├── documents.py     ← POST /documents/index, GET, DELETE
│       ├── query.py         ← POST /query/
│       └── evaluate.py      ← POST /evaluate/
├── frontend/
│   └── app.py               ← Streamlit UI (calls FastAPI via HTTP)
├── ingestion/
│   ├── pdf_extractor.py     ← PyMuPDF, multi-column, metadata
│   └── chunker.py           ← Sentence-aware overlapping chunks
├── embeddings/
│   └── embedder.py          ← SentenceTransformer + LRU cache
├── vector_store/
│   └── chroma_store.py      ← ChromaDB upsert, search, MMR
├── rag/
│   ├── pipeline.py          ← End-to-end RAG orchestrator (Groq)
│   └── prompt_manager.py    ← Versioned prompt templates v1/v2/v3
├── evaluation/
│   └── evaluator.py         ← P@k, Recall@k, MRR, ROUGE-L, faithfulness
├── config.py                ← All parameters (single source of truth)
├── requirements.txt
├── package.json             ← Next.js frontend dependencies
├── start.sh                 ← Launch FastAPI and Streamlit in parallel
└── .env.example
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add: GROQ_API_KEY=gsk_...
```

Start the backend:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Then start the Next.js frontend in a second terminal:

```bash
npm install
npm run dev
```

Alternatively, launch the Streamlit frontend instead:

```bash
streamlit run frontend/app.py
```

## Run Python tests

The backend tests use `pytest` and live under `tests/`. Install the Python
dependencies, then run the complete suite from the repository root:

```bash
pip install -r requirements.txt
python -m pytest
```

Useful focused commands:

```bash
python -m pytest tests/test_chunker.py -v
python -m pytest tests/test_prompt_manager.py::test_prompt_manager_renders_context_and_query
python -m pytest -k schema
```

The tests focus on deterministic behavior and avoid calling Groq, downloading
embedding models, or modifying the persistent ChromaDB store. Run the frontend
checks separately with `npm run lint` and `npm run build`.

Or use the bundled helper script:

```bash
bash start.sh
```

| URL                         | Purpose                          |
| --------------------------- | -------------------------------- |
| http://localhost:8000/docs  | Swagger UI — test every endpoint |
| http://localhost:8000/redoc | ReDoc documentation              |
| http://localhost:3000       | Next.js UI                       |
| http://localhost:8501       | Streamlit UI                     |

## Deployment

This project is best deployed as two connected services:

- **Frontend:** Vercel for the Next.js app in `app/`.
- **Backend:** A Python host such as Render or Fly.io for the FastAPI service.

### Recommended deployment setup

1. Push the repo to GitHub.
2. On Vercel, create a new project from this GitHub repo. Vercel will detect the Next.js app automatically.
3. On Render, create a new Web Service from the same GitHub repo.
   - Environment: Python 3.x
   - Start command: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
   - Set environment variable: `GROQ_API_KEY`
4. In Vercel project settings, add:
   - `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-url>`

### Why this setup?

- Vercel is ideal for the Next.js frontend.
- The FastAPI backend cannot run directly on Vercel as a full Python service, so it needs a dedicated Python host.
- Render/Fly.io supports a persistent web service and can run the FastAPI app with the same repo.

### Local vs production

For local development, keep using:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
npm run dev
```

When deployed, the frontend will call the rendered backend URL via `NEXT_PUBLIC_API_BASE_URL`.

## API Endpoints

| Method   | Path                    | Description                  |
| -------- | ----------------------- | ---------------------------- |
| `GET`    | `/health`               | Liveness + total chunk count |
| `POST`   | `/documents/index`      | Upload & index a PDF         |
| `GET`    | `/documents/`           | List all indexed documents   |
| `DELETE` | `/documents/{filename}` | Remove a document            |
| `POST`   | `/query/`               | RAG query → answer + chunks  |
| `POST`   | `/evaluate/`            | QA evaluation dataset        |

Evaluation uses a citation-source overlap heuristic for faithfulness: it
measures the fraction of retrieved chunk sources that are cited in the answer
(it is not an LLM judge). Reports are saved as JSON in
`evaluation_results/`, and each report records the prompt version used.
Configure
`LLM_INPUT_COST_PER_1M_USD` and `LLM_OUTPUT_COST_PER_1M_USD` to record estimated
token cost. Compare two completed runs from Python:

```python
from evaluation.evaluator import Evaluator
print(Evaluator().compare_runs("run_v1", "run_v2"))
```

Reports include Precision@k, Recall@k, MRR, ROUGE-L, citation-overlap
faithfulness, average latency, total tokens, total estimated cost, and prompt
version, so JSON run reports can be compared to catch regressions when a
versioned prompt changes.

## Groq usage limits

Usage limits depend on the Groq account, model, and plan, and may change.
Check the current limits in the Groq console rather than relying on fixed
values in this README.
