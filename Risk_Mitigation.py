"""Use two LLMs to identify and mitigate risks for a hypothetical support firm."""

import json
import sys
from typing import Any

from financial_forecasting_common import (
    MODEL_CONFIG,
    call_model,
    make_sample_company_data,
    parse_json_object,
)


SYSTEM_PROMPT = (
    "You are a cautious enterprise risk analyst. Base claims on the supplied "
    "data, distinguish observed facts from hypotheses, and avoid invented facts."
)


def validate_risk_response(result: dict[str, Any]) -> dict[str, Any]:
    risks = result.get("risks")
    if not isinstance(risks, list) or not risks:
        raise RuntimeError("Model risk response must contain a non-empty risks list.")
    for item in risks:
        if not isinstance(item, dict):
            raise RuntimeError("Each identified risk must be a JSON object.")
        for field in ("category", "evidence", "impact", "mitigation"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise RuntimeError(f"Risk response is missing string field {field}.")
        likelihood = item.get("likelihood_1_to_5")
        if isinstance(likelihood, bool) or not isinstance(likelihood, int) or not 1 <= likelihood <= 5:
            raise RuntimeError("Risk likelihood must be an integer from 1 to 5.")
    return result


def run() -> int:
    try:
        history = make_sample_company_data()
        latest = history[-1]
        previous = history[-7]
        cash_on_hand = 250000.0
        monthly_fixed_cost = latest["labor_cost_usd"] + latest["cloud_cost_usd"]
        prompt = f"""Assess material financial and operating risks for a hypothetical small technology-support company. All company numbers below are synthetic educational examples, not real firm disclosures. Cash is a stated scenario assumption, not observed data.

Identify market, credit, liquidity, staffing, cyber/service-continuity, and customer-concentration risks where the evidence supports them. Do not assert that the firm has a risk unsupported by the provided facts. For each risk, return its category, observed evidence or explicitly label it as a monitoring gap, impact, likelihood from 1 (low) to 5 (high), and specific mitigation.

Include a rough cash-runway estimate using the stated cash and latest labor/cloud costs; explain that this excludes other costs and collections.

Return only JSON:
{{"risks":[{{"category":"...", "evidence":"...", "impact":"...", "likelihood_1_to_5":3, "mitigation":"..."}}], "cash_runway_months_estimate":0.0, "key_monitoring_gaps":["..."]}}

Latest month: {json.dumps(latest)}
Six months earlier: {json.dumps(previous)}
Recent monthly history: {json.dumps(history[-12:], indent=2)}
Scenario cash on hand USD: {cash_on_hand:.2f}
Latest labor plus cloud costs USD/month: {monthly_fixed_cost:.2f}
"""
        findings = {}
        for key in MODEL_CONFIG:
            findings[key] = validate_risk_response(
                parse_json_object(call_model(key, SYSTEM_PROMPT, prompt))
            )
        result = {
            "company_context": "Hypothetical technology-support business; sample data is synthetic.",
            "sample_data": history,
            "cash_assumption_usd": cash_on_hand,
            "models": {
                key: {"model": MODEL_CONFIG[key]["model"], "analysis": findings[key]}
                for key in MODEL_CONFIG
            },
            "analyst_control_recommendations": [
                "Maintain a rolling 13-week cash forecast and minimum liquidity trigger.",
                "Set customer and receivables concentration limits; review aging weekly.",
                "Use MFA, least privilege, tested offline backups, and incident-response exercises.",
                "Track ticket backlog, SLA breaches, staffing coverage, churn, and cloud spend.",
                "Stress-test wage inflation, client payment delays, and a major customer loss.",
                "Require human review; LLM output is a hypothesis list, not a control or credit decision.",
            ],
            "limitations": (
                "Synthetic data omits debt, actual cash flows, receivables aging, "
                "client concentration, insurance, and security incident history. "
                "Model outputs must be checked against company records."
            ),
        }
        with open("risk_mitigation_results.json", "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        print("Saved risk_mitigation_results.json")
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Risk analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
