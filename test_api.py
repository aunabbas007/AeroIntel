
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


def test_ask_rejects_empty_user_id(client):
    response = client.post(
        "/ask",
        json={
            "question": "Which engine powers the A350-900?",
            "user_id": "",
        },
    )

    assert response.status_code == 422


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
    from unittest.mock import patch
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

    with patch(
        "api.runner.run_async",
        new=fake_run_async,
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
        "Check the server logs for details."
    )
    assert "Unexpected internal test error" not in response.text
