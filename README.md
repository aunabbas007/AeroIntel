# AeroIntel

AeroIntel is a FastAPI aviation assistant. It retrieves evidence from a
Pinecone knowledge base through MCP tools, asks Gemini to write a grounded
answer, and stores conversation history in PostgreSQL.

## What is included

- FastAPI endpoints for health, readiness, login, account creation, questions,
  and conversation history.
- Swagger UI login at `/docs`; users sign in with an email and password.
- Argon2 password hashes and signed, expiring access tokens.
- Gemini primary and fallback models configured through environment variables.
- MCP retrieval with source labels, source-grounded answers, and checks for a
  missing or incompatible Pinecone index.
- Alembic migrations, Docker Compose for local PostgreSQL, and a non-root
  production container.

## Configuration

Copy `.env.example` to `.env` only if `.env` does not already exist. If you
already have a `.env`, preserve it and add or update the required settings
without replacing its secret values.

Required settings:

| Setting | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection URL. For the local Compose stack, use the `db` hostname shown in `.env.example`. |
| `JWT_SECRET_KEY` | Private signing key. Use a random value of at least 32 characters. |
| `GOOGLE_API_KEY` | Gemini API access. |
| `PINECONE_API_KEY` | Pinecone access. |

Optional settings include `ACCESS_TOKEN_EXPIRE_MINUTES` (default 480),
`GEMINI_MODEL_PRIMARY`, `GEMINI_MODEL_FALLBACK`, `PINECONE_INDEX_NAME`,
`PINECONE_CLOUD`, and `PINECONE_REGION`.

Keep `.env` out of Git, Docker images, logs, and chat. On AWS, supply secrets
from AWS Secrets Manager or another managed secret store. Do not put API keys,
JWT signing keys, or database passwords in an image or a source file.

## Run locally with Docker

1. Ensure Docker Desktop is running.
2. Create `.env` from `.env.example` if you do not already have one. Set a
   unique local PostgreSQL password in both `POSTGRES_PASSWORD` and
   `DATABASE_URL`, a random JWT signing key, and your Google and Pinecone keys.
3. Build and start the API and local PostgreSQL service:

   ```powershell
   docker compose -f compose.yaml -f compose.local.yaml up -d --build
   ```

   The container applies pending database migrations before starting the API.
   The API is bound to `127.0.0.1:8000`; it is not exposed to other machines.
4. Create your account once:

   ```powershell
   docker compose -f compose.yaml -f compose.local.yaml exec api python create_user.py
   ```

   The password is entered without being echoed and is stored only as an Argon2
   hash. The command does not print an access token.
   To reset an existing account from the trusted host, add `--reset-password`
   to the same command.
5. Open [http://localhost:8000/docs](http://localhost:8000/docs), select
   **Authorize**, and enter your email in the username field and your password.
   Swagger obtains and stores the access token for you. You do not need to
   generate or paste tokens manually.
6. Check `GET /health`, then `GET /ready`, then ask a question with `POST /ask`.

The initial MCP startup loads the embedding model and connects to Pinecone, so
the first start can take several minutes. The health check allows for this.

## Knowledge base setup

The API expects a Pinecone index whose name matches `PINECONE_INDEX_NAME` and
whose vector dimension is 384. The API fails readiness if the index is missing,
not ready, or has the wrong dimension; it will not silently create an empty
index. To create and populate the index deliberately, configure Pinecone and
run:

```powershell
python index_v2.py
```

This indexes the aviation text files in `data/`. Run indexing as a one-off
operation, not as part of API startup.

## AWS deployment

The repository is prepared to build a container, but this project does not
create AWS resources or change security groups automatically. Keep the existing
RDS instance private. The RDS security group should allow PostgreSQL only from
the AeroIntel application security group. Do not expose port 5432 or the
container's port 8000 to the public internet.

For an EC2 deployment, publish the image to a private ECR repository and run it
with the production values injected from AWS Secrets Manager. The included
`compose.yaml` binds port 8000 to loopback, so an administrator can inspect it
over a restricted SSH tunnel:

```powershell
ssh -i <private-key-file> -L 8000:127.0.0.1:8000 ec2-user@<instance-address>
```

Then open `http://localhost:8000/docs` on the administrator's computer. For a
public application, put an HTTPS load balancer or reverse proxy in front of the
container. Restrict SSH ingress to administrator IP addresses and allow API
ingress only through that HTTPS front end. Never broaden security-group access
to troubleshoot Swagger.

By default, the image runs `alembic upgrade head` before Uvicorn. For a
multi-instance deployment, run `alembic upgrade head` once as a deployment job,
then set `RUN_MIGRATIONS=false` on API tasks so instances do not attempt the
same migration concurrently.

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Process liveness. |
| `GET` | `/ready` | Agent, MCP, and database readiness. |
| `POST` | `/auth/login` | OAuth2 password login used by Swagger. |
| `GET` | `/auth/me` | Return the authenticated account. |
| `POST` | `/ask` | Generate and save a grounded answer. Requires login. |
| `GET` | `/conversations/{session_id}` | Read the signed-in user's conversation history. Requires login. |

Login tokens expire after eight hours by default. Use Swagger's **Authorize**
button to sign in again when needed; no developer-token script is included.

## Troubleshooting

- `401`: sign in again through Swagger's **Authorize** button.
- `/ready` returns `503`: check the sanitized container logs for MCP, Pinecone,
  or database startup failures. `/health` only confirms that the API process is
  responding.
- `429`: the Gemini quota or rate limit is exhausted.
- `503` from `/ask`: an upstream service is unavailable; the configured
  fallback model is attempted by Google ADK.
- “Failed to fetch”: confirm the browser is using the same-origin `/docs` page,
  that `/health` responds, and that the API is reachable through the intended
  secure route. Swagger and `/ask` do not need permissive CORS middleware when
  served by the same FastAPI app.
