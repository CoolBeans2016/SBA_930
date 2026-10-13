"""Compare two Groq models on stock-market and commodity-price analysis.

Data comes from the Federal Reserve Bank of St. Louis FRED public CSV feeds.
The script uses only the Python standard library; provider tokens are read
from the environment or requested with hidden input.
"""

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any

from financial_forecasting_common import (
    GROQ_URL,
    MODEL_CONFIG,
    get_api_key,
    make_sample_company_data,
)


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

SYSTEM_PROMPT = (
    "You are a careful financial-data analyst. Treat the supplied data as the "
    "source of truth, distinguish observations from hypotheses, and do not "
    "invent policy events or data. State uncertainty, assumptions, and "
    "limitations. This is research, not individualized investment advice."
)

SERIES = {
    "SP500": {
        "dataset": "US stock market and macroeconomic indicators",
        "description": "S&P 500 index, daily observations",
        "derived_metric": "monthly_change_pct",
    },
    "FEDFUNDS": {
        "dataset": "US stock market and macroeconomic indicators",
        "description": "Effective federal funds rate, percent",
        "derived_metric": "monthly_change_percentage_points",
    },
    "CPIAUCSL": {
        "dataset": "US stock market and macroeconomic indicators",
        "description": "Consumer Price Index for All Urban Consumers, index",
        "derived_metric": "year_over_year_change_pct",
    },
    "DCOILWTICO": {
        "dataset": "Crude oil prices",
        "description": "West Texas Intermediate spot price, dollars per barrel",
        "derived_metric": "monthly_change_pct",
    },
}

RUBRIC = {
    "accuracy": (
        "Consistency with the supplied observations and correct distinction "
        "between evidence, inference, and uncertain future outcomes"
    ),
    "relevance": (
        "Directly answers both scenarios with useful, data-grounded insights"
    ),
    "reliability": (
        "Acknowledges limits, avoids unsupported certainty, and provides "
        "transparent assumptions and appropriately cautious conclusions"
    ),
}

PROMPT_HISTORY_MONTHS = 12
ASSESSMENT_HISTORY_MONTHS = 9
MAX_COMPLETION_TOKENS = 700
MAX_ASSESSMENT_TOKENS = 1200
MAX_ASSESSED_RESPONSE_CHARS = 1000
GROQ_REQUEST_INTERVAL_SECONDS = 60
_last_groq_request_completed_at: float | None = None


def fetch_fred_series(series_id: str, start_date: date) -> list[dict[str, Any]]:
    """Download and parse one FRED CSV series, retaining observations at/after start_date."""
    request = urllib.request.Request(
        FRED_CSV_URL.format(series_id=series_id),
        headers={"User-Agent": "FinancialForecastingLLMEvaluation/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8-sig")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to download FRED series {series_id}: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "observation_date" not in reader.fieldnames:
        raise RuntimeError(f"FRED response for {series_id} has an unexpected format.")

    observations = []
    for row in reader:
        raw_date = row.get("observation_date", "")
        raw_value = row.get(series_id, "")
        if not raw_date or not raw_value or raw_value == ".":
            continue
        observation_date = date.fromisoformat(raw_date)
        if observation_date < start_date:
            continue
        observations.append(
            {"date": observation_date.isoformat(), "value": float(raw_value)}
        )
    if not observations:
        raise RuntimeError(
            f"FRED returned no usable observations for {series_id} "
            f"on or after {start_date.isoformat()}."
        )
    return observations


def summarize_series(
    series_id: str, observations: list[dict[str, Any]], months: int
) -> dict[str, Any]:
    """Return month-end observations plus the relevant change metric."""
    month_ends: dict[str, dict[str, Any]] = {}
    for observation in observations:
        month_ends[observation["date"][:7]] = observation

    ordered = sorted(month_ends.values(), key=lambda item: item["date"])
    values_by_month = {item["date"][:7]: item["value"] for item in ordered}
    rows = []
    for index, observation in enumerate(ordered):
        row = dict(observation)
        if series_id in ("SP500", "DCOILWTICO") and index > 0:
            previous = ordered[index - 1]["value"]
            row["monthly_change_pct"] = round(
                (observation["value"] / previous - 1) * 100, 2
            )
        elif series_id == "FEDFUNDS" and index > 0:
            row["monthly_change_percentage_points"] = round(
                observation["value"] - ordered[index - 1]["value"], 2
            )
        elif series_id == "CPIAUCSL":
            current_year, current_month = observation["date"][:7].split("-")
            year_ago_month = f"{int(current_year) - 1:04d}-{current_month}"
            if year_ago_month in values_by_month:
                year_ago = values_by_month[year_ago_month]
                row["year_over_year_change_pct"] = round(
                    (observation["value"] / year_ago - 1) * 100, 2
                )
        rows.append(row)

    return {
        "description": SERIES[series_id]["description"],
        "source": f"FRED: {series_id}",
        "units": (
            "index points"
            if series_id in ("SP500", "CPIAUCSL")
            else "percent"
            if series_id == "FEDFUNDS"
            else "US dollars per barrel"
        ),
        "observations": rows[-months:],
        "derived_metric": SERIES[series_id]["derived_metric"],
    }


def fetch_datasets(months: int) -> dict[str, Any]:
    """Fetch public market data and attach the synthetic company dataset."""
    start_date = date.today() - timedelta(days=months * 31 + 400)
    series_data = {}
    for series_id in SERIES:
        print(f"Downloading FRED data: {series_id}...", flush=True)
        raw_data = fetch_fred_series(series_id, start_date)
        series_data[series_id] = summarize_series(series_id, raw_data, months)

    return {
        "hypothetical_tech_support_company": {
            "synthetic": True,
            "source_note": (
                "Deterministic instructional sample data; not actual company "
                "financial or operating results."
            ),
            "observations": make_sample_company_data(months),
        },
        "US stock market and macroeconomic indicators": {
            series_id: series_data[series_id]
            for series_id in ("SP500", "FEDFUNDS", "CPIAUCSL")
        },
        "Crude oil prices": {"DCOILWTICO": series_data["DCOILWTICO"]},
    }


def build_analysis_prompt(datasets: dict[str, Any], policy_context: str) -> str:
    """Create the same two-scenario prompt for both model providers."""
    prompt_datasets = trim_observation_history(datasets, PROMPT_HISTORY_MONTHS)
    return f"""Analyze the supplied data for a hypothetical US technology-support company. The company operating dataset is synthetic teaching data, not a real firm's results. FRED observations are historical public data; the latest available observations may not be from today.

POLICY CONTEXT SUPPLIED BY THE USER:
{policy_context}

SCENARIO 1 — ECONOMIC POLICY AND COMPANY FINANCIAL OUTLOOK
Forecast the potential impact of recent economic policies on the technology-support industry and this hypothetical company's revenue, labor/cloud costs, operating margin, and customer demand over 3–12 months. Use the S&P 500, federal funds rate, CPI, and sample company observations. If the policy context does not identify a verifiable policy and date, say so; do not invent one. The company is private and has no observed stock price, so do not claim to forecast its share price. Explain plausible transmission mechanisms, distinguish correlation from causation, and give a cautious outlook with assumptions and risks.

SCENARIO 2 — COMPANY DEMAND AND OPERATING-COST TRENDS
Analyze the sample ticket-volume, revenue, labor, cloud-cost, churn, and SLA history to forecast 3–12 month company trends. Separately explain how historical WTI crude-oil prices might affect onsite dispatch/travel costs, without treating oil as a direct measure of cloud costs. Give cautious directional forecasts and ranges only where supported, and discuss trend, volatility, assumptions, and material supply/demand risks.

For both scenarios, cite dates and values from the supplied observations where relevant. Separate observed facts from forecasts, state limitations (including that past performance does not guarantee future results), and avoid presenting the output as individualized investment advice.

DATA (JSON):
{json.dumps(prompt_datasets, separators=(",", ":"))}
"""


def trim_observation_history(
    datasets: dict[str, Any], months: int
) -> dict[str, Any]:
    """Keep recent rows in prompts while preserving the full dataset for the report."""
    trimmed: dict[str, Any] = {}
    for dataset_name, dataset in datasets.items():
        if not isinstance(dataset, dict):
            trimmed[dataset_name] = dataset
            continue

        if "observations" in dataset:
            trimmed[dataset_name] = {
                **dataset,
                "observations": dataset["observations"][-months:],
            }
            continue

        trimmed[dataset_name] = {
            series_name: (
                {
                    **series,
                    "observations": series["observations"][-months:],
                }
                if isinstance(series, dict) and isinstance(series.get("observations"), list)
                else series
            )
            for series_name, series in dataset.items()
        }
    return trimmed


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    provider: str,
    api_key: str,
    max_tokens: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps({**payload, "max_tokens": max_tokens}).encode("utf-8"),
        headers={
            **headers,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "my-app/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"{provider} API returned HTTP {exc.code}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{provider} API request failed: {exc}") from exc


def call_model(
    provider: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_COMPLETION_TOKENS,
    reasoning_effort: str | None = None,
) -> str:
    global _last_groq_request_completed_at
    config = MODEL_CONFIG[provider]
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
    payload: dict[str, Any] = {
        "model": config["model"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    response = post_json(
        MODEL_CONFIG[provider]["url"],
        {"Authorization": f"Bearer {api_key}"},
        payload,
        config["label"],
        api_key,
        max_tokens,
    )
    if is_groq_request:
        _last_groq_request_completed_at = time.monotonic()
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"{config['label']} response did not contain message text."
        ) from exc
    if not isinstance(content, str) or not content.strip():
        finish_reason = choice.get("finish_reason", "unknown")
        raise RuntimeError(
            f"{config['label']} returned empty message content "
            f"(finish_reason={finish_reason}); increase the output budget "
            "or simplify the request."
        )
    return content


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a Markdown code fence from a model."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
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
                if not expected_closers or expected_closers.pop() != matching_closers[character]:
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
                raise RuntimeError("LLM evaluator did not return a JSON object.")
            try:
                result = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError("LLM evaluator returned invalid JSON.") from exc
    if not isinstance(result, dict):
        raise RuntimeError("LLM evaluator response must be a JSON object.")
    return result


def build_assessment_prompt(
    model_responses: dict[str, dict[str, str]],
    datasets: dict[str, Any],
) -> str:
    response_text = {
        provider: {
            "model": MODEL_CONFIG[provider]["model"],
            "response": (
                answer
                if len(answer) <= MAX_ASSESSED_RESPONSE_CHARS
                else answer[:700] + "\n...[middle omitted]...\n" + answer[-250:]
            ),
        }
        for provider, answer in model_responses.items()
    }
    rubric_text = "\n".join(
        f"- {criterion}: {description}."
        for criterion, description in RUBRIC.items()
    )
    providers = list(MODEL_CONFIG)
    first, second = providers
    prompt_datasets = trim_observation_history(datasets, ASSESSMENT_HISTORY_MONTHS)
    return f"""Independently assess both model responses against the observed data below. Score each criterion from 1 (poor) to 5 (strong). "Accuracy" here means consistency with the supplied historical observations and sound separation of fact from forecast; future forecast accuracy cannot be known yet. Be evidence-based, note any unsupported claims, and do not reward confidence by itself.

RUBRIC:
{rubric_text}

Return exactly one JSON object and no markdown, code fences, or text before or
after it. Keep each rationale to one short sentence. Example:
{{
  "scores": {{
    "{first}": {{"accuracy": 1, "relevance": 1, "reliability": 1}},
    "{second}": {{"accuracy": 1, "relevance": 1, "reliability": 1}}
  }},
  "rationale": {{
    "{first}": "Brief evidence-based explanation.",
    "{second}": "Brief evidence-based explanation."
  }}
}}

DATA:
{json.dumps(prompt_datasets, separators=(",", ":"))}

RESPONSES:
{json.dumps(response_text, separators=(",", ":"))}
"""


def validate_assessment(parsed: dict[str, Any]) -> dict[str, Any]:
    scores = parsed.get("scores")
    rationale = parsed.get("rationale")
    if not isinstance(scores, dict) or not isinstance(rationale, dict):
        raise RuntimeError("LLM evaluator response is missing scores or rationale.")
    for provider in MODEL_CONFIG:
        if not isinstance(scores.get(provider), dict):
            raise RuntimeError(f"LLM evaluator omitted scores for {provider}.")
        if not isinstance(rationale.get(provider), str):
            raise RuntimeError(f"LLM evaluator omitted rationale for {provider}.")
        for criterion in RUBRIC:
            value = scores[provider].get(criterion)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                raise RuntimeError(
                    f"LLM evaluator gave an invalid {criterion} score for {provider}."
                )
    return {"scores": scores, "rationale": rationale}


def score_summary(assessments: dict[str, Any]) -> dict[str, Any]:
    """Average the two model assessors' scores and report inter-rater disagreement."""
    summary = {}
    for provider in MODEL_CONFIG:
        criterion_means = {}
        for criterion in RUBRIC:
            ratings = [
                assessment["scores"][provider][criterion]
                for assessment in assessments.values()
            ]
            criterion_means[criterion] = round(sum(ratings) / len(ratings), 2)
        criterion_means["mean_score"] = round(
            sum(criterion_means.values()) / len(RUBRIC), 2
        )
        summary[provider] = criterion_means

    disagreements = []
    providers = list(MODEL_CONFIG)
    for candidate in providers:
        for criterion in RUBRIC:
            disagreements.append(
                abs(
                    assessments[providers[0]]["scores"][candidate][criterion]
                    - assessments[providers[1]]["scores"][candidate][criterion]
                )
            )
    summary["assessor_mean_absolute_disagreement"] = round(
        sum(disagreements) / len(disagreements), 2
    )
    return summary


def save_report(output_path: str, report: dict[str, Any]) -> None:
    with open(output_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Groq models analyzing financial data."
    )
    parser.add_argument(
        "--months",
        type=int,
        default=36,
        help="Number of recent month-end observations to include (default: 36).",
    )
    parser.add_argument(
        "--policy-context",
        default=(
            "No specific policy text supplied. Identify policy details only if "
            "they are within your knowledge; state the relevant date and disclose "
            "uncertainty rather than assuming events."
        ),
        help="Policy announcement(s), with dates and source/context if available.",
    )
    parser.add_argument(
        "--output",
        default="llm_evaluation_results.json",
        help="Path to the JSON report (default: llm_evaluation_results.json).",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Fetch data and save the matched prompts without calling LLM APIs.",
    )
    run_mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume rubric assessments from saved model analyses in the output JSON.",
    )
    args = parser.parse_args()
    if args.months < 13:
        parser.error("--months must be at least 13 to calculate year-over-year CPI.")

    try:
        print("Starting financial forecasting LLM evaluation...", flush=True)
        if args.resume:
            with open(args.output, encoding="utf-8") as existing_file:
                report = json.load(existing_file)
            expected_models = {
                provider: config["model"] for provider, config in MODEL_CONFIG.items()
            }
            saved_models = {
                provider: details.get("model")
                for provider, details in report.get("models", {}).items()
            }
            if saved_models != expected_models:
                raise RuntimeError(
                    "Cannot resume: saved result model IDs do not match the "
                    "currently configured model IDs."
                )
            if not isinstance(report.get("responses"), dict) or not all(
                provider in report["responses"] for provider in MODEL_CONFIG
            ):
                raise RuntimeError(
                    "Cannot resume: the output file does not contain analyses "
                    "from both configured models."
                )
            datasets = report["datasets"]
            model_responses = report["responses"]
            print("Loaded saved analyses; resuming rubric assessments.", flush=True)
        else:
            datasets = fetch_datasets(args.months)
            print("FRED data downloaded. Preparing model prompts...", flush=True)
            analysis_prompt = build_analysis_prompt(datasets, args.policy_context)
            report = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "data_sources": [
                    "FRED series SP500, FEDFUNDS, and CPIAUCSL",
                    "FRED series DCOILWTICO",
                ],
                "policy_context": args.policy_context,
                "prompt_history_months": PROMPT_HISTORY_MONTHS,
                "max_completion_tokens_per_request": MAX_COMPLETION_TOKENS,
                "datasets": datasets,
                "prompts": {"analysis": analysis_prompt},
                "models": {
                    provider: {
                        "provider": config["label"],
                        "model": config["model"],
                    }
                    for provider, config in MODEL_CONFIG.items()
                },
            }

        if args.prepare_only:
            report["evaluation_status"] = (
                "Prompts prepared only; no model responses or scores were generated."
            )
        else:
            print(
                "The evaluation makes four API requests (two analyses and "
                "two scoring requests) using the same Groq account. Requests "
                "are spaced by one minute. Progress is saved after each successful "
                "response so a later provider error does not discard completed work.",
                flush=True,
            )
            if not args.resume:
                report["evaluation_status"] = "Model evaluation in progress."
                report["responses"] = {}
                report["peer_assessments"] = {}
                report["assessment_raw_responses"] = {}
                save_report(args.output, report)

                model_responses = {}
                for provider in MODEL_CONFIG:
                    print(
                        f"Requesting analysis from {MODEL_CONFIG[provider]['label']}...",
                        flush=True,
                    )
                    model_responses[provider] = {
                        "analysis": call_model(
                            provider,
                            get_api_key(provider),
                            SYSTEM_PROMPT,
                            analysis_prompt,
                        )
                    }
                    report["responses"] = model_responses
                    report["evaluation_status"] = (
                        f"Analysis received from {provider}; remaining model calls pending."
                    )
                    save_report(args.output, report)
            else:
                report.setdefault("peer_assessments", {})
                report.setdefault("assessment_raw_responses", {})

            assessment_prompt = build_assessment_prompt(model_responses, datasets)
            report["prompts"]["assessment"] = assessment_prompt
            report["evaluation_status"] = "Analyses received; rubric assessments in progress."
            save_report(args.output, report)
            assessments = report["peer_assessments"]
            for assessor in MODEL_CONFIG:
                if assessor in assessments:
                    continue
                saved_assessment = report["assessment_raw_responses"].get(assessor)
                if isinstance(saved_assessment, str) and saved_assessment.strip():
                    try:
                        assessments[assessor] = validate_assessment(
                            parse_json_object(saved_assessment)
                        )
                    except RuntimeError as exc:
                        print(
                            f"Saved assessment from {MODEL_CONFIG[assessor]['label']} "
                            f"could not be recovered ({exc}); requesting it again.",
                            flush=True,
                        )
                    else:
                        print(
                            f"Recovered saved assessment from "
                            f"{MODEL_CONFIG[assessor]['label']}; no API call needed.",
                            flush=True,
                        )
                        report["peer_assessments"] = assessments
                        report["evaluation_status"] = (
                            f"Assessment recovered from saved response for {assessor}; "
                            "remaining scoring calls pending."
                        )
                        save_report(args.output, report)
                        continue
                print(
                    f"Requesting rubric assessment from "
                    f"{MODEL_CONFIG[assessor]['label']}...",
                    flush=True,
                )
                raw_assessment = call_model(
                    assessor,
                    get_api_key(assessor),
                    "You are an impartial evaluator. Return only valid JSON matching the exact structure requested by the user.",
                    assessment_prompt,
                    max_tokens=MAX_ASSESSMENT_TOKENS,
                    reasoning_effort="low",
                )
                report["assessment_raw_responses"][assessor] = raw_assessment
                try:
                    assessments[assessor] = validate_assessment(
                        parse_json_object(raw_assessment)
                    )
                except RuntimeError:
                    report["evaluation_status"] = (
                        f"Assessment from {assessor} could not be parsed or validated."
                    )
                    report["peer_assessments"] = assessments
                    save_report(args.output, report)
                    raise
                report["peer_assessments"] = assessments
                report["evaluation_status"] = (
                    f"Assessment received from {assessor}; remaining scoring calls pending."
                )
                save_report(args.output, report)
            report["peer_assessments"] = assessments
            report["score_summary"] = score_summary(assessments)
            report["evaluation_status"] = "Completed: model responses and rubric assessments saved."
            report["evaluation_limitations"] = (
                "Scores are peer-assessed rubric judgments, not ground-truth "
                "forecast accuracy. Forecast accuracy should be re-evaluated "
                "against actual outcomes after the stated forecast horizon."
            )

        save_report(args.output, report)
        print(f"Saved evaluation report to {args.output}", flush=True)
        if args.prepare_only:
            print("\nMatched analysis prompt:\n")
            print(analysis_prompt)
        else:
            print("\nPeer-assessed scores (1-5):")
            for provider, scores in report["score_summary"].items():
                if provider == "assessor_mean_absolute_disagreement":
                    continue
                print(
                    f"  {provider}: " + ", ".join(
                        f"{criterion}={scores[criterion]}"
                        for criterion in (*RUBRIC, "mean_score")
                    )
                )
            print(
                "  Mean absolute assessor disagreement: "
                f"{report['score_summary']['assessor_mean_absolute_disagreement']}"
            )
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())