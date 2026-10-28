
def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()
    assert data["app"] == "AeroIntel"
    assert data["status"] == "running"
    assert data["message"] == "Welcome to the AeroIntel API"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_rejects_empty_question(client):
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_rejects_missing_question(client):
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_rejects_question_over_2000_characters(client):
    response = client.post(
        "/ask",
        json={"question": "a" * 2001},
    )

    assert response.status_code == 422


def test_ask_ignores_user_id_supplied_in_request(client):
    response = client.post(
        "/ask",
        json={
            "question": "Which engine powers the A350-900?",
            "user_id": "",
        },
    )

    # The API derives identity from the JWT.
    # It must not use the body to override authenticated identity.
    assert response.status_code != 500


from unittest.mock import AsyncMock, patch


def test_ask_returns_429_when_gemini_quota_is_exhausted(client):
    with patch(
        "api.runner.run_async",
        side_effect=Exception(
            "RESOURCE_EXHAUSTED: Gemini API quota exceeded"
        ),
    ):
        response = client.post(
            "/ask",
            json={
                "question": "Which engine powers the A350-900?"
            },
        )

    assert response.status_code == 429
    assert "quota" in response.json()["detail"].lower()




def test_ask_returns_answer_and_session_details(client):
    from unittest.mock import MagicMock, patch
    from google.adk.events import Event
    from google.genai import types


    async def fake_run_async(*args, **kwargs):
        yield Event(
            author="aerointel_agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="The A350-900 uses the Trent XWB-84."
                    )
                ],
            ),
            invocation_id="test-invocation",
            turn_complete=True,
        )

    fake_db = MagicMock()

    with (
        patch("api.runner.run_async", new=fake_run_async),
        patch("api.SessionLocal", return_value=fake_db),
    ):
        response = client.post(
            "/ask",
            json={
                "question": "Which engine powers the A350-900?"
            },
        )

    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "The A350-900 uses the Trent XWB-84."
    assert data["user_id"] == "api_user"
    assert data["session_id"]


def test_ask_returns_502_on_unexpected_agent_error(client):
    from unittest.mock import patch

    async def failing_run_async(*args, **kwargs):
        raise RuntimeError("Unexpected internal test error")
        yield  # Makes this an async generator.

    with patch(
        "api.runner.run_async",
        new=failing_run_async,
    ):
        response = client.post(
            "/ask",
            json={"question": "Tell me about the Airbus A350-900"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "AeroIntel could not process the request. "
        "Check server logs using the request ID."
    )
    assert "Unexpected internal test error" not in response.text

def test_ask_rejects_missing_token(client):
    authorization = client.headers.pop("Authorization", None)
    try:
        response = client.post(
            "/ask",
            json={"question": "Tell me about the A350-900"},
        )
    finally:
        if authorization is not None:
            client.headers["Authorization"] = authorization

    assert response.status_code == 401

def test_ask_rejects_invalid_token(client):
    response = client.post(
        "/ask",
        json={"question": "Tell me about the A350-900"},
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401

from datetime import datetime, timedelta, timezone
import jwt

from source.auth import JWT_SECRET_KEY, JWT_ALGORITHM


def test_ask_rejects_expired_token(client):
    expired_token = jwt.encode(
        {
            "sub": "api_user",
            "exp": datetime.now(timezone.utc) - timedelta(minutes=5),
        },
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )

    response = client.post(
        "/ask",
        json={"question": "Tell me about the A350-900"},
        headers={"Authorization": f"Bearer {expired_token}"},
    )

    assert response.status_code == 401

from unittest.mock import MagicMock, patch


def test_conversation_history_filters_by_authenticated_user(client):
    fake_db = MagicMock()
    fake_db.scalars.return_value.all.return_value = []

    with patch("api.SessionLocal", return_value=fake_db):
        response = client.get("/conversations/test-session")

    assert response.status_code == 200
    assert response.json()["count"] == 0

    statement = fake_db.scalars.call_args.args[0]
    sql = str(statement)

    assert "conversations.session_id" in sql
    assert "conversations.user_id" in sql

def test_repeated_ask_requests_succeed(client):
    from unittest.mock import MagicMock, patch
    from google.adk.events import Event
    from google.genai import types

    async def fake_run_async(*args, **kwargs):
        yield Event(
            author="aerointel_agent",
            content=types.Content(
                role="model",
                parts=[types.Part(text="Aviation test answer.")],
            ),
            invocation_id="repeat-test",
            turn_complete=True,
        )

    fake_db = MagicMock()

    with (
        patch("api.runner.run_async", new=fake_run_async),
        patch("api.SessionLocal", return_value=fake_db),
    ):
        first = client.post(
            "/ask",
            json={"question": "Tell me about the A350-900"},
        )
        second = client.post(
            "/ask",
            json={"question": "Tell me about the Boeing 787"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["answer"] == "Aviation test answer."
    assert second.json()["answer"] == "Aviation test answer."
    assert first.json()["session_id"] != second.json()["session_id"]
    assert fake_db.add.call_count == 2
    assert fake_db.commit.call_count == 2