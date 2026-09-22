import argparse
import asyncio
import json
from pathlib import Path

from wald_agent.cli_errors import safe_cli_error
from wald_agent.config import Settings
from wald_agent.engine import DecisionEngine
from wald_agent.errors import WaldError
from wald_agent.files import example_text
from wald_agent.jev import JevClient
from wald_agent.schemas import DecisionRequest


async def run_decision(path: Path | None, provider: str) -> dict:
    content = (
        await asyncio.to_thread(path.read_text, encoding="utf-8")
        if path
        else example_text("customer_service.json")
    )
    request = DecisionRequest.model_validate_json(content)
    settings = Settings()
    client = JevClient(settings) if provider == "jev" else DecisionEngine(settings)
    async with client:
        return (await client.decide(request)).model_dump(mode="json")


def decide_main() -> None:
    parser = argparse.ArgumentParser(description="Call the Wald LLM engine or Jev directly.")
    parser.add_argument(
        "--input", type=Path, help="Defaults to the packaged customer-service request"
    )
    parser.add_argument("--provider", choices=["llm", "jev"], default="llm")
    args = parser.parse_args()
    try:
        result = asyncio.run(run_decision(args.input, args.provider))
    except (WaldError, OSError, ValueError) as exc:
        parser.exit(2, f"Decision failed: {safe_cli_error(exc)}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    decide_main()
