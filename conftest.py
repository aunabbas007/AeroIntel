
import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import api
from source.auth import create_access_token


@pytest.fixture(scope="module")
def client():
    token = create_access_token("api_user")

    original_runner = api.runner
    original_startup_complete = api.startup_complete

    api.runner = MagicMock()
    api.startup_complete = True

    try:
        test_client = TestClient(api.app)
        test_client.headers.update({
            "Authorization": f"Bearer {token}"
        })
        yield test_client
    finally:
        test_client.close()
        api.runner = original_runner
        api.startup_complete = original_startup_complete
