Report
SBA 930 - Evaluating LLMs for Financial Forecasting and Risk Analysis









Crystal Marie
10/12/2026



Overview	2
Steps	3
Conclusion	4
Appendix	5











Overview
I compared two Groq language models using matched prompts and the same financial information. The analysis used public FRED data for market and economic indicators, together with explicitly labeled synthetic data for a hypothetical technology-support company. I asked the models to analyze financial trends, forecast GDP growth and inflation, and identify financial risks and possible mitigations.









Steps
LLM Evaluation for Financial Forecasting (llm_evaluation_results.json)
Economic Trend Forecasting (trend_forecasting_results.json)
Risk Identification and Mitigation (risk_mitigation_results.json)
Optimization and Enhancement (optimization_enhancement_results.json)


LLM Comparison
I compared openai/gpt-oss-20b and openai/gpt-oss-120b using the same financial data and prompts. Each model’s responses were assessed on a 1–5 scale: accuracy (consistency with the supplied data), relevance (how well the response addressed the task), and reliability (whether claims were cautious and supported).

Data as-of dates: The LLM evaluation was generated October 12, 2026, but its series have different latest observation dates: S&P 500 October 9, federal funds rate September 1, CPI August 1, and WTI October 6.  

Model               Accuracy   Relevance  Reliability   Mean
openai/gpt-oss-20b   3.50/5      3.50/5     3.50/5     3.50/5
openai/gpt-oss-120b  3.00/5      3.50/5     3.00/5     3.17/5


These scores are peer-assessed judgments of the responses, not measured forecast accuracy or proof of future performance. 



Economic Trend Forecasting 
Interpretation:As of the data origin date of April 1, 2026, the two models forecast annualized next-quarter GDP growth of 1.8% and 2.2%, and year-over-year CPI inflation of 3.5% and 3.8%. In the two-period GDP backtest, model 2 had a lower mean absolute error than model 1 (0.71 versus 1.14 percentage points).

Limitation:The GDP backtest included only two periods, so it is too small to establish which model forecasts better over time. The CPI forecasts were not scored, and FRED data may be revised. These results should be treated as preliminary estimates, not reliable predictions.


Risk Identification and Mitigation 
Interpretation:Both models identified overdue invoices and liquidity as concerns. They estimated cash runway at about 1.9 months using the assumed $250,000 cash balance and monthly labor and cloud costs. Suggested actions included monitoring collections and customer concentration, preparing a rolling 13-week cash forecast, and reviewing the models’ findings with a person.

Limitation:The company data is synthetic, and the runway estimate uses only selected costs; it is not a complete cash-flow analysis. The dataset also lacks important details such as actual receivables aging, customer-level revenue, debt, and capital expenses. The models’ risk ratings are suggestions based on the supplied information, not verified assessments of a real company.

Optimization and Enhancement 
I tested whether a more specific prompt improved forecasts over a six-month synthetic holdout. For model 1, the optimized prompt lowered support-ticket MAE from 131.17 to 47.83, but increased revenue MAE from $8,325.96 to $10,200.96. Model 2’s errors were unchanged between its baseline and optimized prompts: revenue MAE was $8,075.96 and ticket MAE was 150.17. Model 1’s optimized prompt had the lowest ticket MAE. Overall, prompt optimization had mixed results in this small synthetic test and did not improve every forecast.


The models received mean peer-assessed rubric scores of 3.50/5 and 3.17/5; these scores reflect judgments about their responses, not verified future forecasting accuracy. For economic forecasting, the models’ GDP growth forecasts were 1.8% and 2.2%, and their CPI inflation forecasts were 3.5% and 3.8%. In a two-period GDP backtest, their mean absolute errors were 1.14 and 0.71 percentage points. In a separate six-month synthetic holdout, prompt optimization improved one model’s support-ticket forecasts but did not improve its revenue forecasts. The naïve last-value baseline had the lowest revenue MAE for both models, while model 1’s optimized prompt had the lowest ticket MAE (47.83 versus 80.83 for the naïve baseline).

The risk analysis highlighted overdue invoices and estimated cash runway of about 1.9 months, based on the report’s assumed $250,000 cash balance and synthetic operating costs. These results should be interpreted cautiously: the company data is synthetic, the GDP backtest is very small, economic data can be revised, and rubric scores are not ground truth. I would use LLMs to support a qualified analyst, not to make financial decisions automatically.

The underlying figures are in: 
Llm_evaluation_results.json
Trend_forecasting_results.json
Risk_mitigation_results.json
optimization_enhancement_results.json


Conclusion
Overall, this project suggests that LLMs can help organize financial information and identify possible trends and risks, but these tests do not establish dependable forecasting performance. Results were mixed, and model outputs should be treated as support for—not a replacement for—human analysis.

The naïve last-value baseline had the lowest revenue MAE for both models. Model 1’s optimized prompt had the lowest ticket MAE (47.83), outperforming the naïve baseline (80.83), but its revenue MAE ($10,200.96) was higher than both its baseline prompt ($8,325.96) and the naïve baseline ($7,149.04). 

The results should be viewed cautiously because the GDP backtest covered only two periods and the company data was synthetic. The risk estimates also relied on assumptions rather than complete company records. Based on these limitations, I conclude that LLMs can support financial analysis by helping organize information and identify questions for further review, but their outputs should be checked against reliable data and reviewed by a qualified person before being used in decision-making.


Appendix

Data sources and methods  
Economic and market data came from the Federal Reserve Bank of St. Louis FRED: real GDP (GDPC1), CPI (CPIAUCSL), the S&P 500 (SP500), the effective federal funds rate (FEDFUNDS), and WTI crude oil (DCOILWTICO). The GDP/CPI forecast origin was April 1, 2026. The separate LLM evaluation was generated October 12, 2026; the latest available observations varied by series: S&P 500, October 9; federal funds rate, September 1; CPI, August 1; and WTI, October 6, 2026. FRED historical data can be revised.

Company operating data was deterministic synthetic teaching data for a hypothetical technology-support business, not real company records. The prompt-optimization comparison used 30 months of training data and a six-month holdout. Forecast errors are reported as MAE and MAPE; lower values indicate smaller errors. The GDP backtest covers two periods, and the CPI forecasts were not scored.

The risk analysis assumed $250,000 in cash. The estimated runway of about 1.9 months used labor and cloud costs only; it is not a complete cash-flow or solvency analysis. Model rubric scores are peer assessments, not objective measures of future forecast accuracy.


Supporting result files: 
LLM evaluation (./llm_evaluation_results.json)
Trend forecasting (./trend_forecasting_results.json)
Risk mitigation (./risk_mitigation_results.json)
Optimization enhancement (./optimization_enhancement_results.json)
