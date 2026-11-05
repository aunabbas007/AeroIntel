from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api
from source.auth import create_access_token, hash_password
from source.database import Base, get_db
from source.models import User


TEST_EMAIL = "api@example.com"
TEST_PASSWORD = "Correct-Horse-Battery-Staple-42!"


@pytest.fixture(scope="module")
def test_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    with factory() as db:
        db.add(
            User(
                id="api_user",
                email=TEST_EMAIL,
                password_hash=hash_password(TEST_PASSWORD),
            )
        )
        db.commit()

    yield factory
    engine.dispose()


@pytest.fixture(scope="module")
def client(test_session_factory):
    original_runner = api.runner
    original_startup_complete = api.startup_complete
    original_lifespan = api.app.router.lifespan_context
    original_session_local = api.SessionLocal

    @asynccontextmanager
    async def test_lifespan(_app):
        yield

    def override_get_db():
        with test_session_factory() as db:
            yield db

    api.app.router.lifespan_context = test_lifespan
    api.app.dependency_overrides[get_db] = override_get_db
    api.SessionLocal = test_session_factory
    api.runner = MagicMock()
    api.startup_complete = True

    token = create_access_token("api_user")
    try:
        with TestClient(api.app) as test_client:
            test_client.headers.update(
                {"Authorization": f"Bearer {token}"}
            )
            yield test_client
    finally:
        api.app.dependency_overrides.clear()
        api.app.router.lifespan_context = original_lifespan
        api.SessionLocal = original_session_local
        api.runner = original_runner
        api.startup_complete = original_startup_complete
