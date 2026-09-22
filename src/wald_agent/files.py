import json
import os
import tempfile
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path


def example_text(name: str) -> str:
    asset = files("wald_agent").joinpath("assets", name)
    if asset.is_file():
        return asset.read_text(encoding="utf-8")
    # Editable installs keep examples at the repository root; wheels embed them.
    root = Path(__file__).resolve().parents[2]
    path = root / "config/.env.example" if name == "env.example" else root / "examples" / name
    return path.read_text(encoding="utf-8")


@contextmanager
def atomic_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".wald-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write(path: Path, content: str) -> None:
    with atomic_text_writer(path) as stream:
        stream.write(content)


def write_json(path: Path, value) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
