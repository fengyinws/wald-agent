import asyncio
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from wald_agent.errors import (
    ConfigurationError,
    IdempotencyConflict,
    InvalidRequest,
    RecordNotFound,
    StorageError,
)

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    principal TEXT NOT NULL,
    kind TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    idempotency_hash TEXT,
    status TEXT NOT NULL CHECK(status IN ('processing', 'succeeded', 'failed')),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT,
    error_json TEXT,
    needs_review INTEGER NOT NULL DEFAULT 0,
    resolution_json TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    UNIQUE(principal, idempotency_hash)
);
CREATE INDEX IF NOT EXISTS decisions_owner_created ON decisions(principal, created_at, id);
CREATE INDEX IF NOT EXISTS decisions_review ON decisions(principal, needs_review, created_at);
"""


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def decode_row(row: sqlite3.Row) -> dict:
    result = dict(row)
    for column, name in (
        ("request_json", "request"),
        ("response_json", "result"),
        ("error_json", "error"),
        ("resolution_json", "resolution"),
    ):
        value = result.pop(column)
        result[name] = json.loads(value) if value is not None else None
    for internal in ("fingerprint", "idempotency_hash", "principal"):
        result.pop(internal, None)
    result["needs_review"] = bool(result["needs_review"])
    return result


class DecisionStore:
    """SQLite WAL storage for one service process; operations run outside the event loop."""

    def __init__(self, path: Path, retention_days: int = 30):
        self.path = path.resolve()
        self.retention_days = retention_days
        self.lock = FileLock(str(self.path) + ".service.lock", thread_local=False)
        self.started = False

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=2.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=2000")
            yield db
        finally:
            db.close()

    async def call(self, function, *args):
        try:
            return await asyncio.to_thread(function, *args)
        except sqlite3.Error as exc:
            raise StorageError("Decision storage is unavailable.") from exc

    async def start(self) -> None:
        await self.call(self._start)

    def _start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.lock.acquire(timeout=0)
        except Timeout as exc:
            raise ConfigurationError(
                "This database is already owned by a service. Run one worker per database."
            ) from exc
        try:
            with self.connection() as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, SCHEMA_VERSION):
                    raise ConfigurationError(
                        "Unsupported database schema version; upgrade the service."
                    )
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=FULL")
                db.executescript(SCHEMA)
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                error = encode(
                    {
                        "code": "request_interrupted",
                        "status_code": 503,
                        "message": (
                            "Service stopped before completion; "
                            "inspect before using a new idempotency key."
                        ),
                    }
                )
                db.execute(
                    "UPDATE decisions SET status='failed', error_json=?, updated_at=? "
                    "WHERE status='processing'",
                    (error, time.time()),
                )
            self.path.chmod(0o600)
            self.started = True
            self._prune()
        except BaseException:
            self.lock.release()
            raise

    async def close(self):
        await asyncio.to_thread(self.lock.release)
        self.started = False

    async def ping(self) -> None:
        def check():
            with self.connection() as db:
                db.execute("SELECT id FROM decisions LIMIT 1").fetchone()

        await self.call(check)

    async def reserve(
        self,
        identifier: str,
        principal: str,
        kind: str,
        fingerprint: str,
        idempotency_hash: str | None,
        request: dict,
    ) -> dict | None:
        return await self.call(
            self._reserve, identifier, principal, kind, fingerprint, idempotency_hash, request
        )

    def _reserve(self, identifier, principal, kind, fingerprint, idempotency_hash, request):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if idempotency_hash is not None:
                row = db.execute(
                    "SELECT * FROM decisions WHERE principal=? AND idempotency_hash=?",
                    (principal, idempotency_hash),
                ).fetchone()
                if row is not None:
                    if row["fingerprint"] != fingerprint:
                        raise IdempotencyConflict(
                            "Idempotency key is already used for a different request."
                        )
                    if row["status"] == "processing":
                        raise IdempotencyConflict(
                            "Request with this idempotency key is still running."
                        )
                    db.commit()
                    return decode_row(row)
            now = time.time()
            db.execute(
                "INSERT INTO decisions(id, principal, kind, fingerprint, idempotency_hash, status, "
                "created_at, updated_at, request_json) VALUES(?,?,?,?,?,'processing',?,?,?)",
                (
                    identifier,
                    principal,
                    kind,
                    fingerprint,
                    idempotency_hash,
                    now,
                    now,
                    encode(request),
                ),
            )
            db.commit()
        return None

    async def finish(self, identifier: str, result: dict | None, error: dict | None):
        def write():
            with self.connection() as db:
                cursor = db.execute(
                    "UPDATE decisions SET status=?, response_json=?, error_json=?, needs_review=?, "
                    "updated_at=? WHERE id=? AND status='processing'",
                    (
                        "failed" if error else "succeeded",
                        encode(result) if result else None,
                        encode(error) if error else None,
                        int(bool(result and result["needs_review"])),
                        time.time(),
                        identifier,
                    ),
                )
                if cursor.rowcount != 1:
                    raise StorageError("Decision completion could not be persisted.")

        await self.call(write)

    async def get(self, identifier: str, principal: str) -> dict:
        def read():
            with self.connection() as db:
                row = db.execute(
                    "SELECT * FROM decisions WHERE id=? AND principal=?", (identifier, principal)
                ).fetchone()
            if row is None:
                raise RecordNotFound("Decision record not found.")
            return decode_row(row)

        return await self.call(read)

    async def list_reviews(
        self, principal: str, resolved: bool, limit: int, after: str | None
    ) -> dict:
        def read():
            with self.connection() as db:
                where = "principal=? AND status='succeeded' AND kind != 'batch'"
                params = [principal]
                where += (
                    " AND resolution_json IS NOT NULL"
                    if resolved
                    else (" AND needs_review=1 AND resolution_json IS NULL")
                )
                if after:
                    cursor = db.execute(
                        "SELECT created_at,id FROM decisions WHERE id=? AND principal=?",
                        (after, principal),
                    ).fetchone()
                    if cursor is None:
                        raise InvalidRequest("Unknown pagination cursor.")
                    where += " AND (created_at>? OR (created_at=? AND id>?))"
                    params.extend([cursor["created_at"], cursor["created_at"], cursor["id"]])
                rows = db.execute(
                    f"SELECT * FROM decisions WHERE {where} ORDER BY created_at,id LIMIT ?",
                    [*params, limit + 1],
                ).fetchall()
            return {
                "items": [decode_row(row) for row in rows[:limit]],
                "next_cursor": rows[limit - 1]["id"] if len(rows) > limit else None,
            }

        return await self.call(read)

    async def resolve(
        self, identifier: str, principal: str, revision: int, resolution: dict
    ) -> dict:
        def write():
            with self.connection() as db:
                cursor = db.execute(
                    "UPDATE decisions SET resolution_json=?, revision=revision+1, updated_at=? "
                    "WHERE id=? AND principal=? AND status='succeeded' AND revision=? "
                    "AND resolution_json IS NULL",
                    (encode(resolution), time.time(), identifier, principal, revision),
                )
                if cursor.rowcount != 1:
                    raise IdempotencyConflict(
                        "Record changed, is incomplete, or has already been reviewed."
                    )

        await self.call(write)
        return await self.get(identifier, principal)

    def _prune(self):
        with self.connection() as db:
            db.execute(
                "DELETE FROM decisions WHERE created_at<? AND status != 'processing'",
                (time.time() - self.retention_days * 86400,),
            )

    async def prune(self):
        await self.call(self._prune)
