import json
import os
import sqlite3
from contextlib import closing

import httpx
import pytest
from conftest import chat_response
from pydantic import ValidationError

from wald_agent.admin import backup_database, doctor, initialize
from wald_agent.cli_errors import safe_cli_error
from wald_agent.config import LOCAL_CONFIG_FILE, Settings
from wald_agent.engine import DecisionEngine
from wald_agent.files import atomic_text_writer, example_text
from wald_agent.sdk import SyncWaldClient
from wald_agent.storage import DecisionStore


def test_initialize_creates_secure_configuration_without_overwriting(tmp_path, capsys):
    initialize(tmp_path)
    private_config = tmp_path / LOCAL_CONFIG_FILE
    settings = Settings(_env_file=private_config)
    secret = settings.wald_api_key.get_secret_value()
    assert settings.wald_env == "production"
    assert len(secret) >= 32
    assert secret not in capsys.readouterr().out
    assert private_config.stat().st_mode & 0o777 == 0o600
    assert private_config.parent.stat().st_mode & 0o777 == 0o700
    public_template = private_config.with_name(".env.example").read_text()
    assert public_template == example_text("env.example")
    assert secret not in public_template
    assert not (tmp_path / ".env").exists()
    assert json.loads((tmp_path / "examples/customer_service.json").read_text())["questions"]
    with pytest.raises(FileExistsError):
        initialize(tmp_path)
    assert Settings(_env_file=private_config).wald_api_key.get_secret_value() == secret


def test_settings_load_private_file_and_environment_takes_precedence(tmp_path, monkeypatch):
    initialize(tmp_path)
    private_config = tmp_path / LOCAL_CONFIG_FILE
    with private_config.open("a") as stream:
        stream.write("\nLLM_API_KEY=private-test-key\nTYPESAFE_API_KEY=private-jev-key\n")
    (tmp_path / ".env").write_text("LLM_API_KEY=legacy-root-key\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().llm_api_key.get_secret_value() == "private-test-key"
    assert Settings().typesafe_api_key.get_secret_value() == "private-jev-key"
    monkeypatch.setenv("LLM_API_KEY", "environment-test-key")
    assert Settings().llm_api_key.get_secret_value() == "environment-test-key"


async def test_online_backup_is_consistent_and_never_overwrites(tmp_path):
    store = DecisionStore(tmp_path / "live.sqlite3")
    await store.start()
    try:
        await store.reserve(
            "test", "client", "decision", "fingerprint", None, {"input": "original"}
        )
        await store.finish("test", {"needs_review": False}, None)
        backup = tmp_path / "backup.sqlite3"
        backup_database(store.path, backup)
        await store.reserve("later", "client", "decision", "fingerprint", None, {})
        with closing(sqlite3.connect(backup)) as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert db.execute("SELECT id FROM decisions").fetchall() == [("test",)]
        with pytest.raises(FileExistsError):
            backup_database(store.path, backup)
        assert backup.stat().st_mode & 0o777 == 0o600
    finally:
        await store.close()


async def test_retention_keeps_active_requests(tmp_path):
    store = DecisionStore(tmp_path / "retention.sqlite3", retention_days=1)
    await store.start()
    try:
        for identifier in ("finished", "active"):
            await store.reserve(identifier, "client", "decision", "fingerprint", None, {})
        await store.finish("finished", {"needs_review": True}, None)
        with store.connection() as db:
            db.execute("UPDATE decisions SET created_at=0")
        await store.prune()
        with store.connection() as db:
            assert db.execute("SELECT id FROM decisions").fetchone()[0] == "active"
            assert db.execute("SELECT count(*) FROM decisions").fetchone()[0] == 1
    finally:
        await store.close()


def test_interrupted_export_does_not_replace_existing_file(tmp_path):
    output = tmp_path / "feedback.jsonl"
    output.write_text("previous\n")
    with pytest.raises(ValueError), atomic_text_writer(output) as stream:
        stream.write("incomplete\n")
        raise ValueError("simulate failed page")
    assert output.read_text() == "previous\n"
    assert list(tmp_path.iterdir()) == [output]


async def test_doctor_checks_without_sending_requests(settings):
    report, ok = await doctor(settings, live=False)
    assert ok
    assert report["providers"]["llm"]["configured"] is True
    assert "test-llm-secret" not in json.dumps(report)
    settings.llm_api_key = None
    report, ok = await doctor(settings, live=False)
    assert not ok
    assert report["providers"]["llm"]["configured"] is False


def test_sync_sdk_reuses_loop_and_pool_then_closes(settings, distributions, decision_request):
    import asyncio

    async def create_response():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json=chat_response(distributions))
            )
        ) as upstream:
            return await DecisionEngine(settings, upstream).decide(decision_request)

    raw = asyncio.run(create_response()).model_dump()
    loops = []

    def handler(request):
        loops.append(asyncio.get_running_loop())
        return httpx.Response(200, json=raw)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with SyncWaldClient(http_client=http) as client:
        assert client.decide(decision_request).answers["department"].choice == "billing"
        client.decide(decision_request)
    assert loops[0] is loops[1]
    with pytest.raises(RuntimeError):
        client.decide(decision_request)
    asyncio.run(http.aclose())


def test_packaged_examples_are_valid_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert json.loads(example_text("customer_service.json"))["state"]
    assert "WALD_API_KEY=" in example_text("env.example")
    assert not os.path.exists("examples")


def test_cli_configuration_error_does_not_echo_secrets():
    with pytest.raises(ValidationError) as failure:
        Settings(
            _env_file=None, llm_api_key="private-key", api_timeout_seconds="secret-invalid-value"
        )
    message = safe_cli_error(failure.value)
    assert "api_timeout_seconds" in message
    assert "private-key" not in message
    assert "secret-invalid-value" not in message
