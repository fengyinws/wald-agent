"""Real HTTP and CLI acceptance test, using a local fake upstream and no paid API."""

import errno
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from wald_agent.sdk import SyncWaldClient

pytestmark = pytest.mark.process


def test_cli_http_persistence_restart_and_feedback_export(tmp_path, request_data):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            answers = {}
            if self.path == "/v1/systemone":
                for name, question in request["questions"].items():
                    kind = question["type"]
                    if kind == "noul":
                        answers[name] = {"type": "noul", "noul": 0.5}
                        continue
                    keys = (
                        list(question["criteria"])
                        if kind == "choice"
                        else [str(i) for i in range(len(question["criteria"]))]
                    )
                    answers[name] = {
                        "type": kind,
                        "confidence": 0.0,
                        "probabilities": {key: 1 / len(keys) for key in keys},
                    }
                    if kind == "choice":
                        answers[name]["choice"] = keys[0]
                    else:
                        answers[name].update(
                            score=(len(keys) - 1) / 2,
                            legend=dict(zip(keys, question["criteria"], strict=True)),
                        )
                response = {"model": "fake-jev-model", "answers": answers}
            else:
                schema = request["response_format"]["json_schema"]["schema"]
                for name, question in schema["properties"]["answers"]["properties"].items():
                    keys = list(question["properties"]["probabilities"]["properties"])
                    answers[name] = {"probabilities": {key: 1 / len(keys) for key in keys}}
                response = {
                    "model": "fake-local-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps({"answers": answers})},
                        }
                    ],
                }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except OSError as error:
        if error.errno in (errno.EPERM, errno.EACCES):
            pytest.skip(
                "Local TCP is blocked by this sandbox; run -m process with socket permission"
            )
        raise
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    key = "local-integration-service-key-123456789"
    environment = os.environ | {
        "WALD_ENV": "production",
        "WALD_API_KEY": key,
        "LLM_API_KEY": "local-upstream-test-key",
        "API_MAX_RETRIES": "0",
        "LLM_BASE_URL": f"http://127.0.0.1:{upstream.server_port}/v1",
        "TYPESAFE_API_KEY": "local-jev-test-key",
        "JEV_BASE_URL": f"http://127.0.0.1:{upstream.server_port}/v1",
        "ENABLED_PROVIDERS": '["llm","jev"]',
        "DATABASE_PATH": str(tmp_path / "service.sqlite3"),
        "NO_PROXY": "*",
    }
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "server.log"

    @contextmanager
    def running():
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "wald_agent.admin", "serve", "--port", str(port)],
                cwd=tmp_path,
                env=environment,
                stdout=log,
                stderr=log,
            )
            try:
                with httpx.Client(base_url=url, timeout=3, trust_env=False) as http:
                    deadline = time.monotonic() + 10
                    while True:
                        try:
                            if http.get("/readyz").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() > deadline or process.poll() is not None:
                            pytest.fail("Service did not start: " + log_path.read_text())
                        time.sleep(0.05)
                    yield http
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    try:
        headers = {"Authorization": f"Bearer {key}", "Idempotency-Key": "real-http-test"}
        with running() as http:
            assert http.post("/v1/decide", json=request_data).status_code == 401
            response = http.post("/v1/decide", json=request_data, headers=headers)
            assert response.status_code == 200, response.text
            result = response.json()
            identifier = result["decision_id"]
            assert result["needs_review"] is True
            assert http.get("/metrics", headers=headers).status_code == 200
            assert "fake-local-model" not in http.get("/metrics", headers=headers).text
            review = http.post(
                f"/v1/reviews/{identifier}",
                headers=headers,
                json={
                    "answers": {"department": "billing", "refund_requested": True, "urgency": 1},
                    "reviewer": "integration-test",
                    "expected_revision": 0,
                },
            )
            assert review.status_code == 200, review.text
            output = tmp_path / "feedback.jsonl"
            exported = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wald_agent.admin",
                    "export-feedback",
                    "--url",
                    url,
                    "--output",
                    str(output),
                ],
                cwd=tmp_path,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert exported.returncode == 0, exported.stderr
            assert json.loads(output.read_text())["expected"]["refund_requested"] is True
        with running() as http:
            replay = http.post("/v1/decide", json=request_data, headers=headers)
            assert replay.json()["decision_id"] == identifier
            with SyncWaldClient(url, key) as client:
                assert client.get_decision(identifier).revision == 1
        assert len(calls) == 1

        def cli(module, *arguments):
            execution = subprocess.run(
                [sys.executable, "-m", module, *arguments],
                cwd=tmp_path,
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert execution.returncode == 0, execution.stderr
            return execution.stdout

        benchmark_path = tmp_path / "benchmark.json"
        cli("wald_agent.benchmark", "--runs", "2", "--warmup", "0", "--output", str(benchmark_path))
        benchmark = json.loads(benchmark_path.read_text())
        assert len(benchmark["records"]) == 2
        assert benchmark["summary"]["wald"]["failures"] == 0
        assert benchmark["summary"]["jev"]["failures"] == 0
        assert benchmark["protocol"]["retries"] == 0
        assert json.loads(cli("wald_agent.cli"))["provider"] == "llm"
        check = json.loads(cli("wald_agent.admin", "doctor", "--live"))
        assert all(item["ok"] for item in check["providers"].values())

        from test_vision import facts_dict

        facts = tmp_path / "facts.json"
        facts.write_text(json.dumps(facts_dict()))
        image = json.loads(cli("wald_agent.vision", "--description-file", str(facts)))
        assert image["native_jev_vision"] is False
        assert image["mode"] == "text_facts_then_jev"
        assert key not in log_path.read_text()
        assert "local-upstream-test-key" not in log_path.read_text()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
