
import os
import sys
from contextlib import asynccontextmanager
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from source.database import SessionLocal
from source.models import Conversation

from fastapi import Depends
from source.auth import get_current_user

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from source.auth import get_current_user

from mcp import StdioServerParameters
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types

load_dotenv()

APP_NAME = "aerointel"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_PATH = os.path.join(BASE_DIR, "mcp_server.py")


# ---------- Request and response models ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None



class AskResponse(BaseModel):
    answer: str
    user_id: str
    session_id: str


# ---------- Shared application services ----------

session_service = InMemorySessionService()
aviation_tools = None
runner = None


# ---------- Application lifecycle ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global aviation_tools, runner

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-u", MCP_SERVER_PATH],
    )

    aviation_tools = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=server_params,
            timeout=60,
        )
    )

    try:
        print("Connecting AeroIntel API to MCP...", flush=True)

        available_tools = await aviation_tools.get_tools()
        print("MCP connection successful!", flush=True)
        print(
            "Available tools:",
            [tool.name for tool in available_tools],
            flush=True,
        )

        agent = LlmAgent(
            name="aerointel_agent",
            model="gemini-3.8-flash",
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

        print("AeroIntel API is ready.", flush=True)
        yield

    finally:
        runner = None

        if aviation_tools is not None:
            await aviation_tools.close()
            aviation_tools = None

        print("AeroIntel API shut down.", flush=True)


# ---------- FastAPI application ----------

app = FastAPI(
    title="AeroIntel API",
    description="Aviation intelligence powered by RAG, MCP, and Google ADK.",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------- Routes ----------

@app.get("/")
async def root():
    return {
        "app": "AeroIntel",
        "status": "running",
        "message": "Welcome to the AeroIntel API",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if runner is not None else "starting"
    }



@app.post("/ask")
async def ask_question(
    request: AskRequest,
    current_user: str = Depends(get_current_user),
):

    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="AeroIntel agent is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question must not be blank.",
        )

    user_id = current_user
    session_id = request.session_id or str(uuid4())

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

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts or []:
                    if part.text:
                        answer_parts.append(part.text)

        answer = "\n".join(answer_parts).strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="The agent returned no text answer.",
            )

        db = SessionLocal()

        try:
            conversation = Conversation(
                user_id=user_id,
                session_id=session_id,
                question=question,
                answer=answer,
            )

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

        except Exception as db_error:
            db.rollback()
            print(
                f"Failed to save conversation: {db_error!r}",
                flush=True,
            )
            raise HTTPException(
                status_code=500,
                detail="The answer was generated, but saving the conversation failed.",
            ) from db_error

        finally:
            db.close()

        return AskResponse(
            answer=answer,
            user_id=user_id,
            session_id=session_id,
        )

    except HTTPException:
        raise

    except Exception as exc:
        error_message = str(exc).lower()
        error_type = type(exc).__name__.lower()

        print(
            f"AeroIntel request failed: {type(exc).__name__}: {exc}",
            flush=True,
        )

        # Gemini quota or rate-limit errors
        quota_indicators = (
            "resource_exhausted",
            "quota exceeded",
            "quota_exceeded",
            "generate_content_free_tier",
            "ratelimitexceeded",
        )

        if (
            any(indicator in error_message for indicator in quota_indicators)
            or "resourceexhausted" in error_type
            or "429" in error_message
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota or rate limit exceeded. "
                    "Check your Google AI Studio quota and retry "
                    "when capacity is available."
                ),
            ) from exc

        # Upstream service unavailable
        if (
            "serviceunavailable" in error_type
            or "503" in error_message
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "The Gemini service is temporarily unavailable. "
                    "Please try again later."
                ),
            ) from exc

        # Other unexpected upstream failures
        raise HTTPException(
            status_code=502,
            detail=(
                "AeroIntel could not process the request. "
                "Check the server logs for details."
            ),
        ) from exc




@app.get("/conversations/{session_id}")
async def get_conversation_history(
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    db = SessionLocal()

    try:
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

    finally:
        db.close()


