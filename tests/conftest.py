from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_ROOT = Path(tempfile.mkdtemp(prefix="forge-lm-tests-"))
os.environ["LLM_PLATFORM_DATABASE_URL"] = f"sqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["LLM_PLATFORM_STORAGE_ROOT"] = ""  # Never inherit the production root from .env.

from llm_platform.main import app  # noqa: E402
from llm_platform.database import engine, init_database  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def migrated_test_database():
    init_database()


@pytest.fixture(autouse=True)
def isolated_metadata(migrated_test_database):
    # Services/background workers open their own sessions, so an outer rollback
    # cannot isolate these tests. Clear only the dedicated test database between
    # cases, preserving the real migrated schema and Alembic revision.
    connection = engine.raw_connection()
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        for (name,) in tables:
            if name != "alembic_version" and not name.startswith("sqlite_"):
                cursor.execute('DELETE FROM "' + name.replace('"', '""') + '"')
        connection.commit()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    finally:
        connection.close()


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


def pytest_sessionfinish(session, exitstatus) -> None:
    del session, exitstatus
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
