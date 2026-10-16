# Architecture

The app runs as one Streamlit process. The active DataFrame lives in the browser session; uploaded bytes are parsed in memory. A local SQLite database stores only facts a user explicitly saves.

```text
CSV/XLSX → loader and profiler → session DataFrame
Question + applicable memories → bounded OpenRouter tool-call loop
Tool arguments → Pydantic validation → pandas analysis functions
Structured results → answer, evidence and expandable trace
```

`dataset/loader.py` checks file type, upload size, row/column limits, sheets and headers. It infers coarse dtypes and semantic roles (for example, numeric measure, identifier, year and multi-value category) while retaining missing values and recording conversions. `tools/analysis.py` contains ten registered calculations; it does not modify the table or accept generated code. Tool results and sampled rows are bounded.

`agent/orchestrator.py` calls OpenRouter's chat completions endpoint with the registered tool schemas and runs validated calls locally. The default `openrouter/free` router selects an available free model that supports tools; the sidebar also lists current zero-priced models that advertise tool support. The model sees the question, schema, relevant saved facts and outputs from requested tools. It does not receive the whole DataFrame by default. Sampled cell values may be sent to the provider.

Memory facts have workspace or exact-dataset scope. Dataset facts take precedence over conflicting workspace facts. Nothing stores chat history or table contents. The SQLite database is local and is not a multi-user shared store.
