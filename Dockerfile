FROM python:3.13-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv==0.12.1
COPY pyproject.toml uv.lock README.md ./
COPY config/.env.example ./config/.env.example
COPY src ./src
COPY examples ./examples
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH" DATABASE_PATH=/app/data/wald.sqlite3 WALD_ENV=production
WORKDIR /app
RUN useradd --create-home --uid 10001 wald && mkdir /app/data && chown wald:wald /app/data
COPY --from=builder /app/.venv /app/.venv
USER wald
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"
CMD ["wald", "serve", "--host", "0.0.0.0", "--port", "8000"]
