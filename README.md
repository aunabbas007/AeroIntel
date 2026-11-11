# AeroIntel

AeroIntel is an aviation knowledge assistant built with FastAPI, Google ADK,
MCP, Groq through LiteLLM, Pinecone, PostgreSQL, Redis, and a browser frontend.
It retrieves evidence from a curated aviation knowledge base, generates
source-grounded answers, and persists authenticated conversation history.

## Architecture

- **Frontend:** static `index.html` served by FastAPI.
- **API:** FastAPI endpoints for registration, login, questions, health,
  readiness, and conversation history.
- **Agent runtime:** Google ADK `LlmAgent` and `Runner`; production model
  configured with `GROQ_MODEL` through LiteLLM.
- **Tool protocol:** MCP server exposes aircraft, airline, and general
  aviation retrieval tools.
- **Retrieval:** Pinecone dense vectors, optional BM25 + dense hybrid search,
  Reciprocal Rank Fusion, and optional cross-encoder reranking.
- **Cache:** Redis retrieval cache with TTL and versioned keys.
- **Persistence:** PostgreSQL with SQLAlchemy and Alembic migrations.
- **Deployment:** Docker, ECR, EC2 Systems Manager, Caddy, and blue-green
  candidate deployment with a retained previous container.

The standalone `agent.py` is a LangGraph retrieval workflow for development
and evaluation. The deployed API currently uses Google ADK in `api.py`.
LangGraph is not silently substituted into the production request path.

## Configuration

Copy `.env.example` to `.env` for local development. Preserve existing
secrets if a local `.env` already exists. Never commit real credentials.

Important settings:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection URL |
| `JWT_SECRET_KEY` | Random signing secret; use at least 32 characters |
| `GROQ_API_KEY` | Production LLM provider credential |
| `GROQ_MODEL` | LiteLLM model identifier |
| `GOOGLE_API_KEY` | Gemini credential for standalone Gemini/LangGraph tools |
| `PINECONE_API_KEY` | Pinecone credential |
| `PINECONE_INDEX_NAME` | Existing 384-dimensional Pinecone index |
| `AEROINTEL_RETRIEVAL_MODE` | `dense` (safe default) or `hybrid` |
| `AEROINTEL_RETRIEVAL_K` | Number of final documents returned |
| `AEROINTEL_RERANKER_ENABLED` | Enable optional cross-encoder reranking |
| `REDIS_CACHE_TTL_SECONDS` | Retrieval cache TTL |
| `AEROINTEL_CACHE_VERSION` | Bump when retrieval output or corpus changes |

The production API uses Groq via LiteLLM. The Google key is retained for the
standalone Gemini/LangGraph experiments and evaluation utilities. Do not assume
a fallback model is active unless it is explicitly configured and tested.

## Local development

1. Install Python 3.11 and Docker Desktop.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create `.env` from `.env.example`, then set local PostgreSQL credentials,
   JWT secret, Groq key, Pinecone key, and index name.
5. Start the API and local database:

   ```bash
   docker compose -f compose.yaml -f compose.local.yaml up -d --build
   ```

6. Create an account:

   ```bash
   docker compose -f compose.yaml -f compose.local.yaml exec api python create_user.py
   ```

7. Open [http://localhost:8000/docs](http://localhost:8000/docs), authorize
   using your email and password, and call `POST /ask`.

The first start may take several minutes because the sentence-transformer
embedding model is downloaded and initialized.

## Knowledge base and Pinecone

The repository includes aviation text documents under `data/aircrafts` and
`data/airlines`. The embedding model
`sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional vectors.

Create/populate the configured index deliberately:

```bash
python index_v2.py
```

Do not run indexing as part of API startup. Confirm the Pinecone index name,
dimension, region, and data population before deploying. Re-index when source
documents change. Keep source metadata such as `type`, `entity_id`, and
`source` intact; retrieval filters and evaluation depend on it.

## Retrieval modes and evaluation

Dense search is the default and preserves current behaviour.

Hybrid mode combines Pinecone dense retrieval with BM25 lexical retrieval,
then fuses rankings using Reciprocal Rank Fusion. The BM25 corpus is rebuilt
from the versioned `data/` documents and cached in-process. It does not require
a new Pinecone index.

To compare retrieval modes against the existing 20-case evaluation set:

```bash
python evaluate_retreival.py
python evaluate_hybrid.py
```

The hybrid comparison reports Precision@K, Recall@K, MRR, and expected-term
coverage for K values 3, 5, and 8. The expected entities and terms in the
evaluation dataset must be reviewed whenever the source corpus changes.

Optional cross-encoder reranking:

```env
AEROINTEL_RERANKER_ENABLED=true
AEROINTEL_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Reranking downloads an additional model and increases CPU/RAM use and latency.
Benchmark it before enabling in production. Hybrid search and reranking remain
configurable; do not switch production mode based on assumptions.

End-to-end answer evaluation against a running local API:

```bash
python evaluate_answers.py
```

The script requires a valid AeroIntel access token and checks expected terms,
abstention behaviour, and latency. It is a smoke/regression evaluator, not a
complete semantic faithfulness benchmark.

## Tests and CI

Run unit/API tests:

```bash
python -m pytest -q
```

Run the focused retrieval tests:

```bash
python -m pytest -q test_rag_retrieval.py
```

The GitHub Actions workflow runs tests, validates Alembic migrations, and
builds the Docker image. Production deployment is restricted to pushes to
`main`; pull requests run validation but do not deploy.

## Production deployment

The existing workflow builds a uniquely tagged image, pushes it to private
ECR, and invokes the EC2 deployment script through Systems Manager. The script
starts Redis, runs database migrations, starts a candidate container on the
inactive loopback port, checks `/ready`, validates Caddy, and switches traffic.
It retains the previous container for rollback.

Before enabling a new release, verify:

- GitHub Actions production environment secrets and AWS permissions.
- EC2 instance is managed by SSM and has least-privilege ECR pull permissions.
- Caddy, Docker, AWS CLI, and required host paths exist on EC2.
- `/etc/aerointel/production.env` contains current, rotated credentials.
- PostgreSQL is reachable privately from EC2 and migrations are backed up.
- Pinecone index exists, is ready, dimension is 384, and contains current data.
- Redis persistence and memory limits are appropriate for the workload.
- HTTPS/DNS and firewall rules expose only the intended public endpoint.
- A rollback procedure has been rehearsed.

Never expose database port 5432, Redis port 6379, or application port 8000
directly to the public internet. Keep production secrets out of GitHub source,
container images, logs, and chat. Prefer AWS-managed secret storage and a
restricted instance role.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process liveness |
| `GET` | `/ready` | Agent, MCP, and database readiness |
| `POST` | `/auth/register` | Create account |
| `POST` | `/auth/login` | Obtain access token |
| `GET` | `/auth/me` | Current authenticated account |
| `POST` | `/ask` | Ask an aviation question |
| `GET` | `/conversations/{session_id}` | Read own conversation history |

## Known operational boundaries

- Production request orchestration currently uses Google ADK, not LangGraph.
- The LangGraph workflow is maintained as a separate experimental/evaluation
  path until it has equivalent behaviour, tests, and a deliberate migration.
- Hybrid retrieval must be benchmarked against dense retrieval on the actual
  Pinecone corpus before it is selected as the production default.
- The answer evaluator checks expected terms and abstention phrases; it does
  not by itself prove factual correctness or faithfulness.
- Production readiness still depends on successful CI, real environment
  validation, security review, backup/restore testing, and a rollback drill.
