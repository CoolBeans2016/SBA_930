"""Forecast macroeconomic indicators relevant to a hypothetical tech-support firm."""

import json
import sys
from datetime import date, timedelta
from typing import Any

from financial_forecasting_common import (
    MODEL_CONFIG,
    call_model,
    fetch_fred_series,
    parse_json_object,
)


def quarter_key(iso_date: str) -> str:
    parsed = date.fromisoformat(iso_date)
    quarter = (parsed.month - 1) // 3 + 1
    return f"{parsed.year}-Q{quarter}"


def quarterly_real_gdp(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the latest GDPC1 observation in each quarter."""
    by_quarter: dict[str, dict[str, Any]] = {}
    for item in observations:
        by_quarter[quarter_key(item["date"])] = item
    return [by_quarter[key] for key in sorted(by_quarter)]


def derive_gdp_growth(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    quarters = quarterly_real_gdp(observations)
    growth = []
    for previous, current in zip(quarters, quarters[1:]):
        annualized = ((current["value"] / previous["value"]) ** 4 - 1) * 100
        growth.append(
            {
                "date": current["date"],
                "quarter": quarter_key(current["date"]),
                "real_gdp_qoq_annualized_pct": round(annualized, 2),
            }
        )
    return growth


def derive_cpi_inflation(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_month = {item["date"][:7]: item for item in observations}
    result = []
    for item in observations:
        current_year, current_month = map(int, item["date"][:7].split("-"))
        prior_month = f"{current_year - 1:04d}-{current_month:02d}"
        if prior_month in by_month:
            yoy = (item["value"] / by_month[prior_month]["value"] - 1) * 100
            result.append(
                {
                    "date": item["date"],
                    "cpi_yoy_pct": round(yoy, 2),
                }
            )
    return result


def make_forecast_prompt(
    origin: str,
    gdp_history: list[dict[str, Any]],
    cpi_history: list[dict[str, Any]],
) -> str:
    return f"""You are forecasting the US economy as context for a hypothetical technology-support company's budgets, wages, cloud contracts, and client demand. The company has no public stock price. Use only data provided below; do not use later knowledge or invent observations.

Forecast from information available on {origin}:
- Real GDP growth for the next quarter, expressed as quarter-over-quarter annualized percent.
- CPI inflation for the next 12 months, expressed as year-over-year percent.

Return only a JSON object with numeric values and a short uncertainty statement:
{{"forecast_gdp_qoq_annualized_pct": 0.0, "forecast_cpi_yoy_pct": 0.0, "uncertainty": "..."}}

Observed GDP growth (FRED GDPC1; calculated from quarterly real GDP levels):
{json.dumps(gdp_history, indent=2)}

Observed CPI inflation (FRED CPIAUCSL; calculated as 12-month percent change):
{json.dumps(cpi_history, indent=2)}
"""


def run() -> int:
    try:
        start = date.today() - timedelta(days=365 * 8)
        gdp = derive_gdp_growth(fetch_fred_series("GDPC1", start))
        cpi = derive_cpi_inflation(fetch_fred_series("CPIAUCSL", start))
        if len(gdp) < 8 or len(cpi) < 14:
            raise RuntimeError("FRED returned too little history for the forecast.")

        backtest_origins = [gdp[-3]["date"], gdp[-2]["date"]]
        evaluations = []
        for origin in backtest_origins:
            origin_quarter = quarter_key(origin)
            origin_index = next(
                index for index, item in enumerate(gdp)
                if item["quarter"] == origin_quarter
            )
            target = gdp[origin_index + 1]
            prompt = make_forecast_prompt(
                origin,
                gdp[max(0, origin_index - 11) : origin_index + 1],
                [item for item in cpi if item["date"] <= origin][-24:],
            )
            row: dict[str, Any] = {
                "origin": origin,
                "actual_next_quarter": target["real_gdp_qoq_annualized_pct"],
                "actual_next_available_cpi_yoy": None,
                "forecasts": {},
            }
            for key in MODEL_CONFIG:
                response = call_model(
                    key,
                    "You are a cautious macroeconomic forecasting analyst. Obey the as-of cutoff and JSON schema.",
                    prompt,
                )
                prediction = parse_json_object(response)
                gdp_forecast = prediction.get("forecast_gdp_qoq_annualized_pct")
                cpi_forecast = prediction.get("forecast_cpi_yoy_pct")
                if not isinstance(gdp_forecast, (int, float)) or isinstance(gdp_forecast, bool):
                    raise RuntimeError(f"{key} returned an invalid GDP forecast.")
                if not isinstance(cpi_forecast, (int, float)) or isinstance(cpi_forecast, bool):
                    raise RuntimeError(f"{key} returned an invalid CPI forecast.")
                row["forecasts"][key] = {
                    "forecast_gdp_qoq_annualized_pct": gdp_forecast,
                    "forecast_cpi_yoy_pct": cpi_forecast,
                    "uncertainty": prediction.get("uncertainty", ""),
                }
            evaluations.append(row)

        current_origin = gdp[-1]["date"]
        current_prompt = make_forecast_prompt(
            current_origin,
            gdp[-12:],
            [item for item in cpi if item["date"] <= current_origin][-24:],
        )
        current = {}
        for key in MODEL_CONFIG:
            prediction = parse_json_object(
                call_model(
                    key,
                    "You are a cautious macroeconomic forecasting analyst. Obey the as-of cutoff and JSON schema.",
                    current_prompt,
                )
            )
            current[key] = prediction

        scores: dict[str, Any] = {}
        for key in MODEL_CONFIG:
            errors = [
                abs(
                    item["forecasts"][key]["forecast_gdp_qoq_annualized_pct"]
                    - item["actual_next_quarter"]
                )
                for item in evaluations
            ]
            scores[key] = {
                "backtest_gdp_mae_percentage_points": round(sum(errors) / len(errors), 2),
                "backtest_periods": len(errors),
                "cpi_forecast_backtest": (
                    "Not scored: a forward 12-month CPI target is unavailable "
                    "for the selected recent origins."
                ),
            }
        result = {
            "company_context": "Hypothetical technology-support company.",
            "data_sources": {
                "GDP": "FRED GDPC1, real GDP, quarterly; growth derived as annualized qoq.",
                "CPI": "FRED CPIAUCSL; inflation derived as 12-month percent change.",
            },
            "data_vintage_limitation": (
                "FRED historical observations may be revised. This test uses "
                "today's revised series, not point-in-time vintages."
            ),
            "backtest": evaluations,
            "backtest_metrics": scores,
            "current_as_of": current_origin,
            "current_forecasts": current,
            "interpretation": (
                "Backtest GDP MAE is an observed score over a small sample, not "
                "proof of future performance. CPI forecasts are reported but "
                "not scored here."
            ),
        }
        with open("trend_forecasting_results.json", "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        print("Saved trend_forecasting_results.json")
        return 0
    except (RuntimeError, OSError, ValueError, StopIteration) as exc:
        print(f"Trend forecasting failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
