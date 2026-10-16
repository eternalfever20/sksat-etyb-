# TableLens

TableLens answers questions about CSV and Excel tables. The language model chooses from registered analysis tools; Python runs each calculation against the loaded table and returns evidence with the answer. It does not generate or execute analysis code.

## Run locally

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Add your OpenRouter key to .env
python -m streamlit run streamlit_app.py
```

If PowerShell blocks activation, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m streamlit run streamlit_app.py`. Upload a CSV/XLSX file or choose the bundled UCI Iris dataset; a synthetic sales table is also available for date and missing-value examples. The Chat tab has five direct-run analysis controls; natural-language questions let the agent choose among all ten tools. Answers show the tool calls and results.

## Stack and tools

Python, Streamlit, pandas, SciPy, Matplotlib, Pydantic, SQLite, python-dotenv, and OpenRouter chat completions with tool calling. `openrouter/free` is the default; route availability and limits can change. Set `OPENROUTER_API_KEY` in `.env` before using the agent. The sidebar refreshes the free tool-capable model list on request.

| Tool | What it computes |
| --- | --- |
| `describe_dataset` | Column types, roles, missing values, distinct counts, examples, and numeric summaries |
| `filter_rows` | Matching rows or counts from one or more conditions |
| `group_aggregate` | Counts and numeric summaries by category |
| `correlation` | Pearson or Spearman relationships between numeric measures |
| `statistical_test` | Welch t-test, one-way ANOVA, or chi-square association |
| `plot_chart` | Histogram, bar, scatter, line, or box chart |
| `detect_outliers` | IQR or z-score outlier candidates |
| `analyze_missingness` | Missing values by column and rows with gaps |
| `trend_over_time` | Counts or aggregates over date/year periods |
| `top_n` | Highest or lowest rows or grouped results |

## Memory

SQLite stores only explicit definitions and preferences, not datasets or transcripts. Facts can be scoped to the workspace or a dataset fingerprint. Relevant facts are matched to later questions; dataset-specific definitions take precedence over workspace-wide ones. Chat-detected definitions require confirmation. The sidebar lets users inspect, update, delete, and clear saved facts.

## Checks

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m evaluation.run_eval
python -m evaluation.run_eval --live  # needs a working OpenRouter key and network
```

The offline evaluation covers the synthetic sales table and real UCI Iris data: profiling, missingness, filtering, aggregation, trends, species comparisons, and correlation. The live check asks the configured model to call tools and return evidence.

## Limits and submission notes

Uploads are limited to 25 MB, 500,000 rows, and 500 columns. Tables and chat stay in the current Streamlit session; charts are temporary local files. Statistical tests assume suitable samples and do not establish causation. The UCI Iris example is real, small, and licensed CC BY 4.0; attribution and the synthetic sales example are documented in `data/README.md`. A generated IMDb-shaped fixture tests year and multi-value-category handling.

Implementation choices represented here are schema-aware dataset loading, bounded local analysis tools, evidence-linked responses, and user-controlled scoped memory. AI assistance was used during implementation and review; before submission, describe which design and review decisions were personally yours. Confirm that the configured GitHub remote is public before submitting.
