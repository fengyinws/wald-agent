import argparse
import asyncio
import json
import os
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from wald_agent.cli_errors import safe_cli_error
from wald_agent.config import LOCAL_CONFIG_FILE, Settings
from wald_agent.engine import DecisionEngine
from wald_agent.errors import ConfigurationError, WaldError
from wald_agent.files import atomic_text_writer, example_text, write_json
from wald_agent.jev import JevClient
from wald_agent.schemas import DecisionRequest
from wald_agent.sdk import WaldClient


def initialize(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    env = directory / LOCAL_CONFIG_FILE
    env.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    template = example_text("env.example")
    content = template.replace("WALD_API_KEY=\n", f"WALD_API_KEY={secrets.token_urlsafe(32)}\n")
    descriptor = os.open(env, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
    template_path = env.with_name(".env.example")
    if not template_path.exists():
        with template_path.open("x", encoding="utf-8") as stream:
            stream.write(template)
    examples = directory / "examples"
    examples.mkdir(exist_ok=True)
    for name in (
        "customer_service.json",
        "customer_service_cases.jsonl",
        "image_questions.json",
        "visual_facts.json",
    ):
        target = examples / name
        if not target.exists():
            with target.open("x", encoding="utf-8") as stream:
                stream.write(example_text(name))
    print(
        f"Created {env.resolve()} with a generated service key. Fill in your upstream credentials."
    )


async def doctor(settings: Settings, live: bool) -> tuple[dict, bool]:
    checks = {}
    for provider in settings.enabled_providers:
        try:
            settings.key_for(provider)
            checks[provider] = {"configured": True}
        except WaldError as exc:
            checks[provider] = {"configured": False, "error": str(exc)}
    result = {
        "environment": settings.wald_env,
        "auth_enabled": bool(settings.service_keys()),
        "providers": checks,
        "live_check": live,
        "database": str(settings.database_path),
    }
    if live:
        request = DecisionRequest.model_validate(
            {
                "state": "A cat is sleeping.",
                "questions": {"cat": {"type": "boolean", "instructions": "Is a cat mentioned?"}},
            }
        )
        for name, check in checks.items():
            if not check["configured"]:
                continue
            client = DecisionEngine(settings) if name == "llm" else JevClient(settings)
            async with client:
                try:
                    response = await client.decide(request)
                    check.update(ok=True, model=response.model, latency_ms=response.latency_ms)
                except WaldError as exc:
                    check.update(ok=False, error=str(exc))
    return result, all(item["configured"] and item.get("ok", True) for item in checks.values())


async def export_feedback(
    settings: Settings, url: str, output: Path, principal: str = "default"
) -> int:
    keys = settings.service_keys()
    if keys and principal not in keys:
        raise ConfigurationError("No service key for this client; select a configured --client.")
    count, cursor = 0, None
    with atomic_text_writer(output) as stream:
        async with WaldClient(url, keys.get(principal, "")) as client:
            while True:
                page = await client.list_reviews(resolved=True, limit=100, after=cursor)
                rows = [
                    {
                        "id": record.id,
                        "request": record.request,
                        "expected": record.resolution["answers"],
                    }
                    for record in page.items
                    if record.kind == "decision" and "state" in record.request
                ]
                chunk = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                await asyncio.to_thread(stream.write, chunk)
                count += len(rows)
                cursor = page.next_cursor
                if cursor is None:
                    break
    return count


def backup_database(source: Path, output: Path):
    if source.resolve() == output.resolve():
        raise ValueError("backup destination must differ from the database")
    if not source.is_file():
        raise FileNotFoundError("database does not exist")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as origin:
            with closing(sqlite3.connect(output)) as destination:
                origin.backup(destination)
                if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("backup integrity check failed")
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Initialize, run, verify and operate Wald.")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init", help="Create private config/.env and sample requests without overwriting files"
    )
    init.add_argument("--directory", type=Path, default=Path("."))
    serve = commands.add_parser("serve", help="Run the authenticated decision service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    check = commands.add_parser(
        "doctor", help="Check local configuration; optionally call real models"
    )
    check.add_argument(
        "--live", action="store_true", help="Send one small paid call per enabled provider"
    )
    check.add_argument("--output", type=Path)
    backup = commands.add_parser("backup", help="Create a consistent SQLite backup while serving")
    backup.add_argument("--output", type=Path, required=True)
    export = commands.add_parser(
        "export-feedback", help="Export reviewed text decisions as benchmark JSONL"
    )
    export.add_argument("--url", default="http://127.0.0.1:8000")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--client", default="default", help="Client name in WALD_API_KEYS")
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize(args.directory)
            return
        settings = Settings()
        if args.command == "serve":
            import uvicorn

            from wald_agent.api import create_app

            settings.validate_startup()
            uvicorn.run(
                create_app(settings),
                host=args.host,
                port=args.port,
                workers=1,
                access_log=False,
                proxy_headers=False,
                log_level=settings.log_level.lower(),
                timeout_graceful_shutdown=int(settings.request_timeout_seconds + 5),
            )
        elif args.command == "doctor":
            report, healthy = asyncio.run(doctor(settings, args.live))
            if args.output:
                write_json(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not healthy:
                raise SystemExit(1)
        elif args.command == "backup":
            backup_database(settings.database_path, args.output)
            print(f"Backup: {args.output.resolve()}")
        elif args.command == "export-feedback":
            count = asyncio.run(export_feedback(settings, args.url, args.output, args.client))
            print(f"Exported {count} reviewed text decisions to {args.output.resolve()}")
    except (WaldError, ValueError, OSError) as exc:
        parser.exit(2, f"Wald: {safe_cli_error(exc)}\n")


if __name__ == "__main__":
    main()
