
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from google.adk.agents import LlmAgent
from google.adk.models import FallbackModel
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters
from pydantic import BaseModel, Field
from sqlalchemy import select, text

from source.auth import get_current_user
from source.database import SessionLocal
from source.models import Conversation


# ============================================================
# 1. CONFIGURATION AND LOGGING
# ============================================================

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("aerointel.api")

APP_NAME = "aerointel"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_PATH = os.path.join(BASE_DIR, "mcp_server.py")

MCP_STARTUP_TIMEOUT = 180

# These describe local application state, not external API quota.
aviation_tools = None
runner = None
mcp_tool_names = []
startup_complete = False


# ============================================================
# 2. REQUEST AND RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    user_id: str
    session_id: str


# ============================================================
# 3. SHARED SERVICES
# ============================================================

session_service = InMemorySessionService()


# ============================================================
# 4. APPLICATION LIFECYCLE AND MCP STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global aviation_tools, runner
    global mcp_tool_names, startup_complete

    startup_started = time.perf_counter()
    startup_complete = False

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", MCP_SERVER_PATH],
        env=os.environ.copy(),
    )

    aviation_tools = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=server_params,
            timeout=MCP_STARTUP_TIMEOUT,
        )
    )

    try:
        logger.info("mcp_startup_started")

        available_tools = await aviation_tools.get_tools()

        mcp_tool_names = [
            tool.name for tool in available_tools
        ]

        if not mcp_tool_names:
            raise RuntimeError(
                "MCP server returned no available tools."
            )

        logger.info(
            "mcp_startup_completed tool_count=%d tools=%s",
            len(mcp_tool_names),
            ",".join(mcp_tool_names),
        )

        agent = LlmAgent(
            name="aerointel_agent",
            model=FallbackModel(
                models=[
                    "gemini-3.8-flash",
                    "gemini-3.6-flash",
                ]
            ),
            description="An aviation intelligence assistant.",
            instruction="""
You are AeroIntel, an aviation intelligence assistant.

Rules:
1. Use MCP tools to retrieve evidence before answering
   factual aviation questions.
2. For aircraft comparisons, retrieve information about
   each aircraft being compared.
3. For airline questions, use the relevant airline tool.
4. Base factual claims on retrieved information.
5. Never invent specifications, figures, or sources.
6. If information is missing, say so clearly.
7. Explain differences clearly using headings or tables
   when appropriate.
""",
            tools=[aviation_tools],
        )

        runner = Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=session_service,
        )

        startup_complete = True

        logger.info(
            "application_ready startup_duration_seconds=%.2f",
            time.perf_counter() - startup_started,
        )

        yield

    except Exception:
        startup_complete = False
        runner = None

        logger.exception("application_startup_failed")
        raise

    finally:
        startup_complete = False
        runner = None
        mcp_tool_names = []

        if aviation_tools is not None:
            try:
                await aviation_tools.close()
                logger.info("mcp_shutdown_completed")
            except Exception:
                logger.exception("mcp_shutdown_failed")
            finally:
                aviation_tools = None

        logger.info("application_shutdown_completed")


# ============================================================
# 5. FASTAPI APPLICATION AND REQUEST CORRELATION
# ============================================================

app = FastAPI(
    title="AeroIntel API",
    description=(
        "Aviation intelligence powered by RAG, MCP, "
        "and Google ADK."
    ),
    version="1.2.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id

    started = time.perf_counter()

    try:
        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "http_request_completed request_id=%s method=%s "
            "path=%s status=%d duration_seconds=%.3f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            time.perf_counter() - started,
        )

        return response

    except Exception:
        logger.exception(
            "http_request_unhandled_exception request_id=%s "
            "method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise


# ============================================================
# 6. HEALTH AND READINESS
# ============================================================

@app.get("/")
async def root():
    return {
        "app": "AeroIntel",
        "status": "running",
        "message": "Welcome to the AeroIntel API",
    }


@app.get("/health")
async def health():
    """Liveness check: confirms the API process responds."""

    return {
        "status": "healthy",
        "app": APP_NAME,
    }


@app.get("/ready")
async def readiness(request: Request):
    """
    Readiness check: verifies local agent initialization
    and database connectivity. Does not call Gemini.
    """

    request_id = request.state.request_id

    if not startup_complete or runner is None:
        logger.warning(
            "readiness_failed request_id=%s reason=agent_not_ready",
            request_id,
        )

        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {
                    "agent": "not_ready",
                    "database": "not_checked",
                },
                "request_id": request_id,
            },
        )

    db = None

    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))

        return {
            "status": "ready",
            "checks": {
                "agent": "ready",
                "mcp": "connected",
                "database": "connected",
            },
            "mcp_tool_count": len(mcp_tool_names),
            "request_id": request_id,
        }

    except Exception:
        logger.exception(
            "readiness_failed request_id=%s reason=database_unavailable",
            request_id,
        )

        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "checks": {
                    "agent": "ready",
                    "database": "unavailable",
                },
                "request_id": request_id,
            },
        )

    finally:
        if db is not None:
            db.close()


# ============================================================
# 7. ERROR CLASSIFICATION
# ============================================================

def classify_upstream_error(exc: Exception) -> int:
    """Map upstream failures to appropriate HTTP status codes."""

    error_message = str(exc).lower()
    error_type = type(exc).__name__.lower()

    quota_indicators = (
        "resource_exhausted",
        "resourceexhausted",
        "quota exceeded",
        "quota_exceeded",
        "generate_content_free_tier",
        "ratelimitexceeded",
        "rate limit exceeded",
        "too many requests",
    )

    if (
        any(item in error_message for item in quota_indicators)
        or any(item in error_type for item in quota_indicators)
        or "429" in error_message
    ):
        return 429

    unavailable_indicators = (
        "serviceunavailable",
        "service unavailable",
        "temporarily unavailable",
    )

    if (
        any(item in error_message for item in unavailable_indicators)
        or "serviceunavailable" in error_type
        or "503" in error_message
    ):
        return 503

    return 502


def raise_upstream_http_error(
    exc: Exception,
    request_id: str,
) -> None:
    status_code = classify_upstream_error(exc)

    if status_code == 429:
        detail = (
            "Gemini API quota or rate limit exceeded. "
            "Check model quota and retry when capacity is available."
        )
        category = "upstream_quota_exhausted"

    elif status_code == 503:
        detail = (
            "An upstream service is temporarily unavailable. "
            "Please try again later."
        )
        category = "upstream_unavailable"

    else:
        detail = (
            "AeroIntel could not process the request. "
            "Check server logs using the request ID."
        )
        category = "upstream_failure"

    logger.error(
        "request_failed request_id=%s category=%s "
        "exception_type=%s",
        request_id,
        category,
        type(exc).__name__,
    )

    raise HTTPException(
        status_code=status_code,
        detail=detail,
        headers={"X-Request-ID": request_id},
    ) from exc


# ============================================================
# 8. ASK ENDPOINT
# ============================================================

@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    http_request: Request,
    current_user: str = Depends(get_current_user),
):
    request_id = http_request.state.request_id
    request_started = time.perf_counter()

    if not startup_complete or runner is None:
        logger.warning(
            "ask_rejected request_id=%s reason=agent_not_ready",
            request_id,
        )

        raise HTTPException(
            status_code=503,
            detail="AeroIntel agent is not ready.",
            headers={"X-Request-ID": request_id},
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be blank.",
            headers={"X-Request-ID": request_id},
        )

    user_id = current_user
    session_id = request.session_id or str(uuid4())

    logger.info(
        "ask_started request_id=%s question_length=%d "
        "session_is_new=%s",
        request_id,
        len(question),
        request.session_id is None,
    )

    try:
        existing_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        if existing_session is None:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )

        message = types.Content(
            role="user",
            parts=[types.Part(text=question)],
        )

        answer_parts = []
        event_count = 0
        runner_started = time.perf_counter()

        logger.info(
            "agent_execution_started request_id=%s",
            request_id,
        )

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            event_count += 1

            if event.is_final_response() and event.content:
                for part in event.content.parts or []:
                    if part.text:
                        answer_parts.append(part.text)

        runner_duration = time.perf_counter() - runner_started

        logger.info(
            "agent_execution_completed request_id=%s "
            "duration_seconds=%.3f event_count=%d",
            request_id,
            runner_duration,
            event_count,
        )

        answer = "\n".join(answer_parts).strip()

        if not answer:
            logger.error(
                "empty_agent_response request_id=%s",
                request_id,
            )

            raise HTTPException(
                status_code=502,
                detail="The agent returned no text answer.",
                headers={"X-Request-ID": request_id},
            )

        # Persist the generated answer.
        db = None
        persistence_started = time.perf_counter()

        try:
            db = SessionLocal()

            conversation = Conversation(
                user_id=user_id,
                session_id=session_id,
                question=question,
                answer=answer,
            )

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            logger.info(
                "conversation_saved request_id=%s "
                "duration_seconds=%.3f",
                request_id,
                time.perf_counter() - persistence_started,
            )

        except Exception as db_error:
            if db is not None:
                try:
                    db.rollback()
                except Exception:
                    logger.exception(
                        "database_rollback_failed request_id=%s",
                        request_id,
                    )

            logger.exception(
                "conversation_save_failed request_id=%s "
                "exception_type=%s",
                request_id,
                type(db_error).__name__,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "The answer was generated, but saving "
                    "the conversation failed."
                ),
                headers={"X-Request-ID": request_id},
            ) from db_error

        finally:
            if db is not None:
                db.close()

        total_duration = time.perf_counter() - request_started

        logger.info(
            "ask_completed request_id=%s "
            "duration_seconds=%.3f",
            request_id,
            total_duration,
        )

        return AskResponse(
            answer=answer,
            user_id=user_id,
            session_id=session_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "ask_execution_failed request_id=%s "
            "exception_type=%s",
            request_id,
            type(exc).__name__,
        )

        raise_upstream_http_error(exc, request_id)


# ============================================================
# 9. CONVERSATION HISTORY
# ============================================================

@app.get("/conversations/{session_id}")
async def get_conversation_history(
    session_id: str,
    request: Request,
    current_user: str = Depends(get_current_user),
):
    request_id = request.state.request_id
    db = None

    try:
        db = SessionLocal()

        statement = (
            select(Conversation)
            .where(
                Conversation.session_id == session_id,
                Conversation.user_id == current_user,
            )
            .order_by(
                Conversation.created_at.asc(),
                Conversation.id.asc(),
            )
        )

        conversations = db.scalars(statement).all()

        logger.info(
            "conversation_history_loaded request_id=%s count=%d",
            request_id,
            len(conversations),
        )

        return {
            "session_id": session_id,
            "count": len(conversations),
            "conversations": [
                {
                    "id": conversation.id,
                    "user_id": conversation.user_id,
                    "question": conversation.question,
                    "answer": conversation.answer,
                    "created_at": (
                        conversation.created_at.isoformat()
                        if conversation.created_at
                        else None
                    ),
                }
                for conversation in conversations
            ],
        }

    except Exception as exc:
        logger.exception(
            "conversation_history_failed request_id=%s "
            "exception_type=%s",
            request_id,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve conversation history.",
            headers={"X-Request-ID": request_id},
        ) from exc

    finally:
        if db is not None:
            db.close()
