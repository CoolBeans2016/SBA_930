"""Shared sample data and LLM-provider helpers for the SBA 930 scripts."""

import csv
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from typing import Any


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_REQUEST_INTERVAL_SECONDS = 60
MAX_COMPLETION_TOKENS = 1200
_last_groq_request_completed_at: float | None = None

MODEL_CONFIG = {
    "groq_model_1": {
        "label": "Groq model 1",
        "token_env": "GROQ_API_KEY",
        "url": GROQ_URL,
        "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
    },
    "groq_model_2": {
        "label": "Groq model 2",
        "token_env": "GROQ_API_KEY",
        "url": GROQ_URL,
        "model": os.environ.get(
            "GROQ_MODEL_2", "openai/gpt-oss-120b"
        ),
    },
}

_API_KEYS: dict[str, str] = {}


def get_api_key(provider: str) -> str:
    """Read a provider token from the environment or visible terminal input."""
    config = MODEL_CONFIG.get(provider)
    if config is None:
        raise RuntimeError(f"Unknown model provider: {provider}")

    token_env = config["token_env"]
    api_key = os.environ.get(token_env) or _API_KEYS.get(token_env)
    if not api_key:
        api_key = input(f"{config['label']} token for {token_env}: ").strip()
    if not api_key:
        raise RuntimeError(
            f"A token in {token_env} is required to use {config['label']}."
        )
    _API_KEYS[token_env] = api_key
    return api_key




def make_sample_company_data(months: int = 36) -> list[dict[str, Any]]:
    """Build a deterministic, explicitly synthetic tech-support business dataset."""
    if months < 2:
        raise ValueError("At least two months are required.")

    end_month = date.today().replace(day=1)
    first_month = (end_month - timedelta(days=months * 31)).replace(day=1)
    rows = []
    for index in range(months):
        month_number = first_month.month - 1 + index
        year = first_month.year + month_number // 12
        month = month_number % 12 + 1
        month_date = date(year, month, 1)
        seasonal = math.sin((month - 1) * 2 * math.pi / 12)
        tickets = round(1800 + index * 24 + seasonal * 85)
        revenue = round(142000 + index * 2100 + seasonal * 4500, 2)
        labor_cost = round(79000 + index * 1050 + seasonal * 1600, 2)
        cloud_cost = round(11800 + index * 230, 2)
        other_operating_cost = round(21500 + index * 280, 2)
        rows.append(
            {
                "month": month_date.isoformat(),
                "revenue_usd": revenue,
                "support_tickets": tickets,
                "labor_cost_usd": labor_cost,
                "cloud_cost_usd": cloud_cost,
                "other_operating_cost_usd": other_operating_cost,
                "operating_margin_pct": round(
                    (revenue - labor_cost - cloud_cost - other_operating_cost)
                    / revenue
                    * 100,
                    2,
                ),
                "sla_met_pct": round(min(98.2, 92.5 + index * 0.12 + seasonal * 0.4), 2),
                "customer_churn_pct": round(max(1.4, 3.8 - index * 0.045 - seasonal * 0.15), 2),
                "overdue_invoice_share_pct": round(8.5 + (index % 5) * 0.7, 2),
            }
        )
    return rows


def fetch_fred_series(series_id: str, start_date: date) -> list[dict[str, Any]]:
    """Download observations from FRED's public CSV endpoint."""
    request = urllib.request.Request(
        FRED_CSV_URL.format(series_id=series_id),
        headers={"User-Agent": "FinancialForecastingSBA930/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read().decode("utf-8-sig")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to download FRED series {series_id}: {exc}") from exc

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or "observation_date" not in reader.fieldnames:
        raise RuntimeError(f"FRED response for {series_id} has an unexpected format.")

    observations = []
    for row in reader:
        raw_date = row.get("observation_date", "")
        raw_value = row.get(series_id, "")
        if not raw_date or not raw_value or raw_value == ".":
            continue
        observation_date = date.fromisoformat(raw_date)
        if observation_date >= start_date:
            observations.append(
                {"date": observation_date.isoformat(), "value": float(raw_value)}
            )
    if not observations:
        raise RuntimeError(
            f"FRED returned no usable observations for {series_id} "
            f"on or after {start_date.isoformat()}."
        )
    return observations


def call_model(provider: str, system_prompt: str, user_prompt: str) -> str:
    """Call one configured chat-completions model and return its text response."""
    global _last_groq_request_completed_at
    config = MODEL_CONFIG[provider]
    api_key = get_api_key(provider)
    is_groq_request = config["url"] == GROQ_URL
    if is_groq_request and _last_groq_request_completed_at is not None:
        elapsed = time.monotonic() - _last_groq_request_completed_at
        wait_seconds = GROQ_REQUEST_INTERVAL_SECONDS - elapsed
        if wait_seconds > 0:
            print(
                f"Waiting {wait_seconds:.0f}s for Groq's per-minute token limit...",
                flush=True,
            )
            time.sleep(wait_seconds)
    payload = {
        "model": config["model"],
        "temperature": 0.2,
        "max_tokens": MAX_COMPLETION_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if is_groq_request:
        payload["reasoning_effort"] = "low"
    request = urllib.request.Request(
        config["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "my-app/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"{config['label']} API returned HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{config['label']} API request failed: {exc}") from exc

    if is_groq_request:
        _last_groq_request_completed_at = time.monotonic()
    try:
        choice = result["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"{config['label']} response did not contain message text."
        ) from exc
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        finish_reason = choice.get("finish_reason", "unknown")
        usage = result.get("usage", {})
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        message_fields = ", ".join(sorted(message)) if isinstance(message, dict) else "unknown"
        raise RuntimeError(
            f"{config['label']} returned empty message content "
            f"(finish_reason={finish_reason}, completion_tokens={completion_tokens}, "
            f"message_fields={message_fields})."
        )
    return content


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON response, including one wrapped in a Markdown code fence."""
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Model returned empty response content instead of JSON.")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as original_error:
        expected_closers = []
        in_string = False
        escaped = False
        matching_closers = {"}": "{", "]": "["}
        for character in cleaned:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character in "{[":
                expected_closers.append(character)
            elif character in "}]":
                if (
                    not expected_closers
                    or expected_closers.pop() != matching_closers[character]
                ):
                    expected_closers = []
                    break
        if not in_string and expected_closers:
            repaired = cleaned.rstrip() + "".join(
                "}" if opener == "{" else "]"
                for opener in reversed(expected_closers)
            )
            try:
                result = json.loads(repaired)
            except json.JSONDecodeError:
                result = None
        else:
            result = None

        if result is None:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end < start:
                excerpt = " ".join(cleaned[:240].split())
                raise RuntimeError(
                    "Model did not return a JSON object. Response begins: "
                    f"{excerpt!r}"
                ) from original_error
            try:
                result = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Model returned invalid JSON at "
                    f"line {exc.lineno}, column {exc.colno}: {exc.msg}. "
                    f"Response excerpt: {' '.join(cleaned[:240].split())!r}"
                ) from exc
    if not isinstance(result, dict):
        raise RuntimeError("Model response must be a JSON object.")
    return result
