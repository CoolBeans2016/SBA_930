# SBA 930: A Student's Guide

## The project in one sentence

This project compares two language models on financial questions for a
hypothetical technology-support business, then checks economic forecasts, risks,
and one prompt-improvement idea against clearly identified data.

## What the data means

- **Company operations:** the scripts create 36 months of synthetic sample
  revenue, support tickets, labor/cloud/other costs, service-level performance,
  customer churn, and overdue invoices. These are practice values, not actual
  records from a real company.
- **Economic context:** the scripts download US macroeconomic and market series
  from the Federal Reserve Bank of St. Louis FRED. Public data is not necessarily
  real-time and may be revised.
- **The company is private:** it has no observed share price in this exercise.
  The analysis considers how economic conditions might affect its sales, costs,
  customer demand, and margins instead of pretending to predict a private stock.
- **Two models:** the defaults are Groq `openai/gpt-oss-20b` and
  `openai/gpt-oss-120b`. Set `GROQ_MODEL` and `GROQ_MODEL_2` to choose other
  models. Both models use the same Groq API account but have different model IDs.

## What each Python file does

### `LLM_Evaluation.py` — compare forecasts and analysis

Downloads S&P 500, federal funds rate, CPI, and WTI crude-oil data, then adds
the sample company's operating history. It sends the same carefully scoped
prompt with the latest 12 monthly observations to each model and asks both
models to score the responses against the latest nine months for accuracy,
relevance, and reliability. The saved report retains the full downloaded
history. Response lengths are capped to help stay within provider token limits.
Groq calls are spaced by one minute to reduce the risk of exceeding the
account's per-minute token limit. Scoring requests use low reasoning effort and
a larger output budget so the model has room to return the requested JSON.

The scores are rubric judgments made by LLMs, not proof that a future forecast
was right. Future forecast accuracy is only measurable after the forecast
horizon has passed and can be compared with actual results.

### `Trend_Forecasting.py` — forecast GDP and inflation

Uses FRED real GDP (`GDPC1`) and CPI (`CPIAUCSL`). It calculates GDP's
quarter-over-quarter annualized growth and CPI's year-over-year inflation.
It asks the two models to forecast these indicators from an explicit
historical cutoff, then scores GDP forecasts for two recent quarters using
mean absolute error (MAE).

There are only two GDP backtest periods, and CPI forecasts are not scored
because a complete future 12-month outcome is not available for the recent
forecast dates. The GDP results use today's downloaded data, which can include
revisions made after the forecast origin.

### `Risk_Mitigation.py` — find risks and suggest controls

Asks both models to identify risks that are supported by the sample data or
clearly labeled as information gaps. It covers market, credit, liquidity,
staffing, customer, and service/cyber risks. It also asks for an illustrative
cash-runway estimate based on an explicitly stated cash assumption.

The runway estimate is not a full solvency analysis: it excludes collections,
debt, taxes, and many other cash flows. A human analyst must verify each risk
against real records.

### `Optimization_Enhancement.py` — test an improved prompt

Holds out the last six sample months, asks each model for the same six-month
forecast with a basic and an improved prompt, and compares each forecast with
the held-out revenue and ticket counts. It also compares with a simple
last-observed-value forecast.

MAE and mean absolute percentage error (MAPE) summarize forecast errors; lower
is better. This synthetic holdout comes from the same generated pattern as the
training data, so good scores here do not demonstrate accuracy on real
companies or markets.

### `financial_forecasting_common.py` — shared building blocks

Keeps the synthetic data generator, FRED download helper, model settings,
terminal token prompt, and API call code in one place so the project
does not repeat them in every script.

## How to run

Use a VS Code integrated Terminal opened in this project folder. The scripts
use only the Python standard library; no package installation is needed.
They require internet access for FRED and a Groq API key. The scripts request
a missing key using visible terminal input, or you can set it in PowerShell
before running:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
# Optional: choose the Groq chat-completion models.
$env:GROQ_MODEL = "openai/gpt-oss-20b"
$env:GROQ_MODEL_2 = "openai/gpt-oss-120b"
```

Typed tokens are visible in the terminal; avoid screenshots or sharing the
terminal while entering them. Never paste tokens into a Python file, report,
chat, or Git commit. If a token is exposed, revoke it and create a replacement.

```powershell
python .\LLM_Evaluation.py --prepare-only
python .\LLM_Evaluation.py
python .\LLM_Evaluation.py --resume
python .\Trend_Forecasting.py
python .\Risk_Mitigation.py
python .\Optimization_Enhancement.py
```

The first command prepares matched prompts and writes
`llm_evaluation_results.json` without calling a model. The other commands make
model/API calls and save JSON outputs beside the scripts. Groq calls are
spaced by one minute to reduce the risk of exceeding the account's
per-minute token limit:

If evaluation stops after both analyses have been saved, rerun with
`python .\LLM_Evaluation.py --resume` to continue scoring without repeating
the analysis calls. Resume also reuses any saved scoring response it can
validate, repairing only missing JSON closing delimiters. Scoring prompts
request a compact JSON object; the script validates the returned object and
scores locally.

- `llm_evaluation_results.json`
- `trend_forecasting_results.json`
- `risk_mitigation_results.json`
- `optimization_enhancement_results.json`

The model-running scripts use API calls, which may incur charges. The key is
not written into the result files. Before submission, inspect the JSON outputs
and transfer the relevant data, findings, charts/tables, and limitations into
the official Per Scholas report template.

## Terms to explain to your instructor

- **Forecast:** a reasoned estimate about a future value; it is not a fact.
- **Backtest:** simulate a forecast using only data available before a past
  target period, then compare it with what happened.
- **MAE:** the average absolute difference between prediction and actual value,
  in the original unit (for example, dollars of revenue).
- **MAPE:** the average absolute error as a percent of actual values. It is
  easier to compare across scales, but is not useful when actual values are
  zero or near zero.
- **Synthetic data:** invented sample values, useful for demonstrating a
  method but not evidence about a real company.
- **Risk mitigation:** actions that reduce the likelihood or impact of a risk,
  such as cash forecasting, receivables monitoring, security controls, and
  operational contingency planning.

## A simple explanation you can give

“I used two language-model versions and gave them the same historical
information, so I could compare their answers fairly. I used public FRED data
for the economy and market, and clearly labeled sample operating data for a
hypothetical tech-support company. I checked the historical consistency of
answers and used a small holdout test with MAE/MAPE for numerical forecasts.
The results are limited: the company data is synthetic, the backtest is small,
LLM rubric scores are not ground truth, and economic data can be revised. I
would use the models to help an analyst, not make decisions automatically.”

## Responsible-use checklist

1. Label sample data, dates, units, sources, and assumptions.
2. Separate observed facts from model-generated interpretations and forecasts.
3. Do not treat correlation as proof that a policy caused a market move.
4. Check model claims and calculations against source data; models can
   hallucinate, omit risks, or sound certain while being wrong.
5. Protect customer and company confidential information; do not send it to an
   external model without authorization.
6. Require qualified human review before financial, credit, staffing, or
   customer-impacting decisions. This class project is not investment advice.
