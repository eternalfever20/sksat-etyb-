"""TableLens Streamlit interface."""

from __future__ import annotations

import html
import importlib
import io
import json
import logging
import os
import re
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import agent.orchestrator as orchestrator_module
import dataset.loader as loader_module
import dataset.profiler as profiler_module
import memory.store as memory_module
import tools.analysis as analysis_module

# Streamlit can rerun an edited script in a process that still has older imported
# modules cached. Refresh the local project modules when their API is outdated.
if (
    getattr(orchestrator_module, "APP_API_VERSION", 0) != 4
    or len(analysis_module.TOOL_SCHEMAS) != 10
):
    importlib.reload(analysis_module)
    importlib.reload(orchestrator_module)
    importlib.reload(loader_module)
    importlib.reload(profiler_module)
    importlib.reload(memory_module)

DEFAULT_MODEL = orchestrator_module.DEFAULT_MODEL
AgentError = orchestrator_module.AgentError
ask = orchestrator_module.ask
available_free_tool_models = orchestrator_module.available_free_tool_models
test_connection = orchestrator_module.test_connection
DatasetError = loader_module.DatasetError
load_dataset = loader_module.load_dataset
profile_dataset = profiler_module.profile_dataset
MemoryStore = memory_module.MemoryStore
parse_plain_definition = memory_module.parse_plain_definition

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

TOOL_GUIDANCE = {
    "describe_dataset": (
        "Table overview",
        "Inspect types, semantic roles, missing values, distinct counts, and examples.",
    ),
    "filter_rows": (
        "Find rows",
        "Filter real records with conditions, sorting, and selected columns.",
    ),
    "group_aggregate": (
        "Group and summarize",
        "Compare totals, averages, or counts across categories.",
    ),
    "correlation": ("Correlation", "Measure how numeric columns move together."),
    "statistical_test": (
        "Statistical test",
        "Compare groups or test association between categories.",
    ),
    "plot_chart": ("Chart", "Create a chart from the loaded data."),
    "detect_outliers": ("Possible outliers", "Flag unusual numeric values for review."),
    "analyze_missingness": (
        "Missing data",
        "Show missingness by column and rows with missing data.",
    ),
    "trend_over_time": (
        "Trend",
        "Summarize rows or a numeric measure over dates or years.",
    ),
    "top_n": ("Rank results", "Rank rows or aggregated groups by a chosen measure."),
}


@st.cache_data(ttl=1800, show_spinner=False)
def free_model_options() -> list[dict[str, str]]:
    return available_free_tool_models()


try:
    streamlit_secrets = st.secrets
except (FileNotFoundError, AttributeError):
    streamlit_secrets = {}
for setting_name in (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "MAX_UPLOAD_MB",
    "AGENT_MAX_STEPS",
    "WORKSPACE_ID",
    "MEMORY_DB_PATH",
    "LOG_LEVEL",
):
    if setting_name not in os.environ:
        try:
            secret_value = streamlit_secrets.get(setting_name)
        except FileNotFoundError:
            secret_value = None
        if secret_value is not None:
            os.environ[setting_name] = str(secret_value)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
)
st.set_page_config(
    page_title="Data Analyst",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="auto",
)
st.markdown(
    """<style>
.block-container{max-width:1480px;padding:2rem 3rem 2.5rem}
[data-testid="stSidebar"]>div:first-child{padding-top:1.25rem}
.brand{font-size:1.05rem;font-weight:650;margin:0 0 2rem}
.section-label{font-size:.83rem;font-weight:550;margin:1.25rem 0 .55rem}
.dataset-card{border:1px solid rgba(90,80,60,.18);border-radius:8px;padding:14px 15px;margin:.25rem 0 .6rem}
.dataset-name,.dataset-title{font-weight:650;overflow-wrap:anywhere}
.dataset-meta,.dataset-stats{font-size:.88rem;margin-top:4px}
.model-line{display:flex;align-items:center;gap:9px;font-family:monospace;font-size:.82rem;overflow-wrap:anywhere}
.status-dot{width:9px;height:9px;background:#2e8b68;border-radius:50%;display:inline-block;flex:none}
.status-dot.missing{background:#b7791f}
.user-row{display:flex;justify-content:flex-end;margin:1.25rem 0 .9rem}
.user-bubble{max-width:min(76%,720px);padding:12px 15px;border:1px solid rgba(90,80,60,.18);border-radius:10px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
[data-testid="stButton"] button{white-space:normal;height:auto;min-height:44px}
.empty-title{font-size:1.9rem;font-weight:650;margin-bottom:.4rem}
.empty-copy{max-width:660px;font-size:1rem;line-height:1.6;margin-bottom:1.4rem}
@media(max-width:900px){.block-container{padding:1.25rem 1rem}.user-bubble{max-width:92%}}
</style>""",
    unsafe_allow_html=True,
)


def store() -> MemoryStore:
    return MemoryStore(
        os.getenv("MEMORY_DB_PATH", "./data/memory.sqlite3"),
        os.getenv("WORKSPACE_ID", "local-default"),
    )


def clear_dataset() -> None:
    for key in (
        "frame",
        "info",
        "chat_history",
        "last_result",
        "last_question",
        "question",
        "initial_upload",
        "pending_question",
        "quick_tool_result",
    ):
        st.session_state.pop(key, None)


def upload_limit() -> int:
    raw = os.getenv("MAX_UPLOAD_MB", "25")
    try:
        limit = int(raw)
    except ValueError as exc:
        raise DatasetError(
            "MAX_UPLOAD_MB must be a whole number from 1 to 25."
        ) from exc
    if not 1 <= limit <= 25:
        raise DatasetError("MAX_UPLOAD_MB must be between 1 and 25.")
    return limit


def accept_upload(upload, sheet: str | None = None) -> None:
    if upload is None:
        return
    try:
        limit = upload_limit()
        data = upload.getvalue()
        if len(data) > limit * 1024 * 1024:
            raise DatasetError(f"File exceeds the {limit} MB upload limit.")
        frame, info = load_dataset(upload.name, data, limit, sheet)
    except DatasetError as exc:
        st.error(str(exc))
        return
    info["size_bytes"] = len(data)
    previous = st.session_state.get("info", {}).get("fingerprint")
    if previous and previous != info["fingerprint"]:
        st.session_state.chat_history = []
        st.session_state.pop("quick_tool_result", None)
    st.session_state.frame = frame
    st.session_state.info = info


def suggestions_for(types: dict[str, str], roles: dict[str, str]) -> list[str]:
    numeric = [c for c, role in roles.items() if role == "numeric_measure"]
    groups = [
        c
        for c, role in roles.items()
        if role in {"category", "multi_value_category", "boolean"}
    ]
    time = [c for c, role in roles.items() if role in {"date", "year"}]
    suggestions = ["Describe columns", "Find missing values"]
    if len(numeric) > 1:
        suggestions.append(f"Correlate {numeric[0]} and {numeric[1]}")
    if numeric:
        suggestions.append(f"Summarize {numeric[0]}")
        suggestions.append(f"Plot {numeric[0]}")
    if numeric and groups:
        suggestions.append(f"Compare {numeric[0]} by {groups[0]}")
    if time:
        suggestions.append(f"Trend by {time[0]}")
    elif groups:
        suggestions.append(f"Count values in {groups[0]}")
    first_column = next(iter(types), None)
    if first_column:
        suggestions.append(f"Show rows with {first_column}")
    return list(dict.fromkeys(suggestions))[:6]


def running_label(name: str, arguments: dict) -> str:
    columns = []
    for key in (
        "column",
        "numeric_column",
        "metric",
        "group_by",
        "time_column",
        "x_column",
        "columns",
    ):
        value = arguments.get(key)
        if isinstance(value, list):
            columns.extend(str(column) for column in value)
        elif value:
            columns.append(str(value))
    for condition in arguments.get("conditions", []):
        if isinstance(condition, dict) and condition.get("column"):
            columns.append(str(condition["column"]))
    if name == "analyze_missingness" and not columns:
        columns.append("all columns")
    context = ", ".join(dict.fromkeys(columns))
    return f"Running {name} on {context}…" if context else f"Running {name}…"


def render_evidence(item: dict) -> None:
    result = item["result"]
    name = item["name"]
    friendly_name, explanation = TOOL_GUIDANCE.get(name, (name, "Analysis result."))
    args = json.dumps(item["arguments"], ensure_ascii=False, separators=(",", ": "))
    st.markdown(
        f'<div class="evidence-head"><span>{html.escape(friendly_name)} · '
        f"<code>{html.escape(name)}</code></span>"
        f'<span class="evidence-args">{html.escape(args)}</span></div>',
        unsafe_allow_html=True,
    )

    if not result.get("ok"):
        st.caption(explanation)
        st.error(result.get("error", {}).get("message", "The analysis tool failed."))
        return

    st.caption(explanation)
    st.caption(f"Evidence ID · {result['evidence_id']}")
    view = {
        "profile": "Column profile",
        "groups": "Results",
        "pairs": "Column pairs",
        "columns": "Missing values",
        "periods": "Periods",
        "rows": "Rows",
        "sample_rows": "Example rows",
    }
    shown = set()
    if name == "plot_chart" and result.get("file_path"):
        st.image(
            result["file_path"],
            caption=f"{result.get('chart', 'Data')} chart",
            use_container_width=True,
        )
    for key, label in view.items():
        values = result.get(key)
        if isinstance(values, list) and values:
            st.caption(label)
            st.dataframe(pd.DataFrame(values), width="stretch", hide_index=True)
            shown.add(key)
    if name == "filter_rows":
        st.caption(
            f"{result.get('matched_rows', 0):,} matching rows · {result.get('returned_rows', 0):,} shown"
        )
    if name == "analyze_missingness":
        st.caption(f"Rows with missing data · {result.get('rows_with_missing', 0):,}")
    if name in {"statistical_test", "correlation"}:
        summary = {
            key: value
            for key, value in result.items()
            if key not in {"ok", "evidence_id", "pairs"}
        }
        st.json(summary)
    if result.get("note"):
        st.caption(result["note"])
    if not shown and name not in {"plot_chart", "filter_rows"}:
        st.json(
            {
                key: value
                for key, value in result.items()
                if key not in {"ok", "evidence_id"}
            }
        )
    with st.expander("Calculation details"):
        st.caption(f"Evidence ID · {result.get('evidence_id', 'unavailable')}")
        st.code(
            json.dumps(item["arguments"], ensure_ascii=False, indent=2, default=str),
            language="json",
        )
        st.caption(f"Tool: {name} · {explanation}")


def render_memory(info: dict | None) -> None:
    with st.expander("Saved definitions", expanded=False):
        try:
            memory_store = store()
            saved = memory_store.list_all()
        except (OSError, sqlite3.Error) as exc:
            st.error(f"Saved definitions are unavailable: {exc}")
            return
        st.caption("Saved locally. Tables and chat transcripts are not stored.")
        editing = st.selectbox(
            "Edit a saved definition",
            ["New definition"]
            + [f"{item['kind']} · {item['key']} · {item['id']}" for item in saved],
            key="memory_edit_choice",
        )
        selected = next(
            (item for item in saved if editing.endswith(f"· {item['id']}")), None
        )
        with st.form("memory_form"):
            left, right = st.columns([1, 2])
            kind_options = [
                "metric_definition",
                "column_definition",
                "filter",
                "preference",
                "correction",
            ]
            kind = left.selectbox(
                "Fact type",
                kind_options,
                index=kind_options.index(selected["kind"])
                if selected and selected["kind"] in kind_options
                else 0,
                key=f"memory_kind_{selected['id'] if selected else 'new'}",
            )
            edit_id = selected["id"] if selected else "new"
            key = right.text_input(
                "Subject / key",
                value=selected["key"] if selected else "",
                placeholder="net revenue",
                key=f"memory_key_{edit_id}",
            )
            value = st.text_area(
                "Reusable definition",
                value=selected["value"] if selected else "",
                placeholder="Revenue after refunds; exclude returned rows",
                key=f"memory_value_{edit_id}",
            )
            workspace_wide = st.checkbox(
                "Apply across datasets in this workspace",
                value=bool(selected and selected["dataset_fingerprint"] is None),
                key=f"memory_scope_{selected['id'] if selected else 'new'}",
            )
            save = st.form_submit_button(
                "Update definition" if selected else "Save definition", type="primary"
            )
        if save:
            try:
                if (
                    not workspace_wide
                    and info is None
                    and not (selected and selected["dataset_fingerprint"] is not None)
                ):
                    raise ValueError(
                        "Load a dataset before saving a dataset-scoped definition."
                    )
                scope = (
                    None
                    if workspace_wide
                    else (
                        selected["dataset_fingerprint"]
                        if selected and selected["dataset_fingerprint"] is not None
                        else info["fingerprint"]
                    )
                )
                saved_item = memory_store.upsert(
                    {"kind": kind, "key": key, "value": value}, scope
                )
                if selected and saved_item["id"] != selected["id"]:
                    memory_store.delete(selected["id"])
                scope_label = "workspace" if workspace_wide else "this dataset"
                action = "updated" if saved_item["updated"] else "saved"
                st.success(f"Definition {action} for {scope_label}.")
                st.rerun()
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(f"Could not save definition: {exc}")
        if saved:
            for memory in saved:
                left, right = st.columns([5, 1])
                scope_label = (
                    "workspace-wide"
                    if memory["dataset_fingerprint"] is None
                    else f"dataset {memory['dataset_fingerprint'][:10]}"
                )
                left.markdown(
                    f"**{memory['kind']} · {memory['key']}** · {scope_label}  \n{memory['value']}"
                )
                if right.button("Delete", key=f"delete_memory_{memory['id']}"):
                    memory_store.delete(memory["id"])
                    st.rerun()
            if st.button("Clear all workspace memories"):
                count = memory_store.clear()
                st.success(f"Deleted {count} saved definitions.")
                st.rerun()
        else:
            st.caption("No saved definitions for this workspace and dataset yet.")


def render_quick_tools(df: pd.DataFrame, info: dict) -> None:
    """Let users run common tools directly; chat remains the agent workflow."""
    tool_labels = {
        "Column profile": "describe_dataset",
        "Filter rows": "filter_rows",
        "Group summary": "group_aggregate",
        "Correlation": "correlation",
        "Missing data": "analyze_missingness",
    }
    with st.expander("Run a tool directly", expanded=True):
        choice = (
            st.segmented_control(
                "Choose an analysis",
                list(tool_labels),
                default="Column profile",
                key="quick_tool_choice",
                label_visibility="collapsed",
            )
            or "Column profile"
        )
        roles = info.get("roles", {})
        categories = [
            column
            for column, role in roles.items()
            if role in {"category", "multi_value_category", "boolean", "identifier"}
        ]
        numeric = [
            column for column, role in roles.items() if role == "numeric_measure"
        ]
        arguments = {}
        can_run = True

        with st.form("quick_tool_form"):
            tool_name = tool_labels[choice]
            if tool_name == "filter_rows":
                column = st.selectbox(
                    "Column", list(df.columns), key="quick_filter_column"
                )
                operator = st.selectbox(
                    "Condition",
                    [
                        "equals",
                        "contains",
                        "greater than",
                        "less than",
                        "is missing",
                        "is present",
                    ],
                    key="quick_filter_operator",
                )
                value = None
                if operator not in {"is missing", "is present"}:
                    if roles.get(column) == "boolean":
                        value = st.selectbox(
                            "Value", [True, False], key="quick_filter_value"
                        )
                    else:
                        value = st.text_input("Value", key="quick_filter_value")
                operators = {
                    "equals": "eq",
                    "contains": "contains",
                    "greater than": "gt",
                    "less than": "lt",
                    "is missing": "is_null",
                    "is present": "not_null",
                }
                condition = {"column": column, "operator": operators[operator]}
                if value is not None:
                    condition["value"] = value
                arguments = {"conditions": [condition], "limit": 20}
            elif tool_name == "group_aggregate":
                if categories:
                    group_column = st.selectbox(
                        "Group by", categories, key="quick_group_column"
                    )
                    metrics = ["Count rows"] + numeric
                    metric_choice = st.selectbox(
                        "Measure", metrics, key="quick_group_metric"
                    )
                    agg_choice = st.selectbox(
                        "Calculation",
                        ["count", "mean", "sum", "median"],
                        key="quick_group_agg",
                    )
                    arguments = {
                        "group_by": [group_column],
                        "metric": None
                        if metric_choice == "Count rows"
                        else metric_choice,
                        "agg": "count" if metric_choice == "Count rows" else agg_choice,
                    }
                else:
                    can_run = False
                    st.caption("This tool needs a categorical or boolean column.")
            elif tool_name == "correlation":
                if len(numeric) >= 2:
                    left, right = st.columns(2)
                    first = left.selectbox(
                        "First measure", numeric, key="quick_corr_left"
                    )
                    others = [column for column in numeric if column != first]
                    second = right.selectbox(
                        "Second measure", others, key="quick_corr_right"
                    )
                    arguments = {"columns": [first, second], "method": "pearson"}
                else:
                    can_run = False
                    st.caption("This tool needs at least two numeric measure columns.")

            submitted = st.form_submit_button(
                "Run analysis", type="primary", disabled=not can_run
            )
            if submitted:
                if (
                    tool_name == "describe_dataset"
                    or tool_name == "analyze_missingness"
                ):
                    arguments = {}
                result = analysis_module.run_tool(
                    tool_name,
                    df,
                    info["types"],
                    arguments,
                    roles=roles,
                    conversions=info.get("conversions", {}),
                )
                st.session_state.quick_tool_result = {
                    "name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "fingerprint": info["fingerprint"],
                }

    quick_result = st.session_state.get("quick_tool_result")
    if quick_result and quick_result.get("fingerprint") == info["fingerprint"]:
        st.caption("Direct tool run · computed locally from the loaded table")
        render_evidence(quick_result)


with st.sidebar:
    st.markdown('<div class="brand">Data Analyst</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
    loaded = "frame" in st.session_state

    if loaded:
        info = st.session_state.info
        size_kb = info.get("size_bytes", 0) / 1024
        st.markdown(
            '<div class="dataset-card">'
            f'<div class="dataset-name">▤ &nbsp;{html.escape(info["name"])}</div>'
            f'<div class="dataset-meta">{size_kb:,.1f} KB · {len(st.session_state.frame):,} × {len(st.session_state.frame.columns)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        replace_col, remove_col = st.columns([1, 1])
        with replace_col:
            replacement = st.file_uploader(
                "Replace",
                type=["csv", "xlsx", "xlsm"],
                key=f"replace_upload_{info['fingerprint'][:10]}",
                label_visibility="collapsed",
                help="Choose a replacement CSV or Excel workbook.",
            )
        with remove_col:
            if st.button("Remove", key="remove_dataset", width="stretch"):
                clear_dataset()
                st.rerun()
        if replacement is not None:
            selected_sheet = None
            if replacement.name.lower().endswith((".xlsx", ".xlsm")):
                try:
                    workbook = pd.ExcelFile(io.BytesIO(replacement.getvalue()))
                    if len(workbook.sheet_names) > 1:
                        selected_sheet = st.selectbox(
                            "Workbook sheet",
                            workbook.sheet_names,
                            key=f"sheet_{replacement.file_id}",
                        )
                except (ValueError, OSError, ImportError, zipfile.BadZipFile) as exc:
                    st.error(f"Could not inspect workbook sheets: {exc}")
            accept_upload(replacement, selected_sheet)
    else:
        st.caption("None loaded")
        example_tables = {
            "Iris flowers · UCI real data": "iris.csv",
            "Sales orders · synthetic example": "demo_sales.csv",
        }
        selected_example = st.selectbox(
            "Example table", list(example_tables), key="example_table"
        )
        if st.button("Load example table", key="load_demo", width="stretch"):
            try:
                example_name = example_tables[selected_example]
                example_data = (ROOT / "data" / example_name).read_bytes()
                frame, info = load_dataset(example_name, example_data, upload_limit())
                info["size_bytes"] = len(example_data)
                st.session_state.frame = frame
                st.session_state.info = info
                st.session_state.chat_history = []
                st.session_state.pop("quick_tool_result", None)
                st.rerun()
            except (OSError, DatasetError, ValueError) as exc:
                st.error(f"Could not load the demo: {exc}")

    api_configured = bool(os.getenv("OPENROUTER_API_KEY"))
    configured_model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    free_models = st.session_state.get("available_free_models", [])
    if st.button("Refresh free model list", width="stretch"):
        with st.spinner("Checking OpenRouter free routes…"):
            free_models = free_model_options()
            st.session_state.available_free_models = free_models
        if not free_models:
            st.info(
                "No free tool-capable routes were returned. The automatic route remains available."
            )
    model_ids = [DEFAULT_MODEL] + [
        item["id"] for item in free_models if item["id"] != DEFAULT_MODEL
    ]
    model_labels = {DEFAULT_MODEL: "Automatic free route"}
    model_labels.update({item["id"]: f"{item['name']} · free" for item in free_models})
    if configured_model not in model_ids:
        model_ids.append(configured_model)
        model_labels[configured_model] = f"Configured model · {configured_model}"
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Choose a model",
        model_ids,
        index=model_ids.index(configured_model),
        format_func=lambda value: model_labels[value],
        key="openrouter_model_choice",
        help="The automatic option picks a free model that supports tool calls. Other listed choices are currently free and advertise tool support.",
    )
    st.caption("Free model availability and usage limits can change.")
    status_class = "" if api_configured else " missing"
    st.markdown(
        f'<div class="model-line"><span class="status-dot{status_class}"></span>'
        f"{html.escape(selected_model)}</div>",
        unsafe_allow_html=True,
    )
    if not api_configured:
        st.caption("Add OPENROUTER_API_KEY to .env to ask questions.")
    if st.button("Test connection", disabled=not api_configured, width="stretch"):
        try:
            connection_result = test_connection(selected_model)
            st.success(connection_result)
        except AgentError as exc:
            st.error(str(exc))
    st.markdown(
        '<div class="section-label">Analysis tools</div>', unsafe_allow_html=True
    )
    st.caption(
        "Run five common analyses directly in Chat, or ask a question to use all ten tools."
    )
    render_memory(st.session_state.get("info"))
    st.divider()
    if st.button("Clear conversation", key="clear_conversation", width="stretch"):
        st.session_state.chat_history = []
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_question", None)
        st.rerun()


if "frame" not in st.session_state:
    _, upload_column, _ = st.columns([0.55, 2, 0.55])
    with upload_column:
        st.markdown(
            '<div class="empty-title">Upload a table to begin</div>'
            '<div class="empty-copy">Ask questions in plain language. Every answer shows '
            "the calculations and rows behind it.</div>",
            unsafe_allow_html=True,
        )
        main_upload = st.file_uploader(
            "Upload a CSV or Excel file",
            type=["csv", "xlsx", "xlsm"],
            key="initial_upload",
            help="Up to 25 MB, 500,000 rows, and 500 columns.",
        )
        if main_upload is not None:
            selected_sheet = None
            if main_upload.name.lower().endswith((".xlsx", ".xlsm")):
                try:
                    workbook = pd.ExcelFile(io.BytesIO(main_upload.getvalue()))
                    if len(workbook.sheet_names) > 1:
                        selected_sheet = st.selectbox(
                            "Workbook sheet",
                            workbook.sheet_names,
                            workbook.sheet_names[0],
                            key=f"sheet_{main_upload.file_id}",
                        )
                except (ValueError, OSError, ImportError, zipfile.BadZipFile) as exc:
                    st.error(f"Could not inspect workbook sheets: {exc}")
            accept_upload(main_upload, selected_sheet)
            if "frame" in st.session_state:
                st.rerun()
        st.caption("Your file is read in this session and is not stored.")
    st.stop()


df = st.session_state.frame
info = st.session_state.info
types = info["types"]
prof = profile_dataset(df, types, info.get("roles", {}), info.get("conversions", {}))
missing_cells = int(df.isna().sum().sum())
st.markdown(
    f'<div class="dataset-title">{html.escape(info["name"])}</div>',
    unsafe_allow_html=True,
)
if info["name"] == "iris.csv":
    st.caption(
        "Iris dataset · Fisher (1936), from the "
        "[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) · CC BY 4.0"
    )
missing_class = "stat-warn" if missing_cells else ""
st.markdown(
    f'<div class="dataset-stats">{len(df):,} rows · {len(df.columns):,} columns · '
    f"{info['duplicate_rows']:,} duplicate rows · "
    f'<span class="{missing_class}">{missing_cells:,} missing cells</span></div>',
    unsafe_allow_html=True,
)

ask_tab, data_tab = st.tabs(["Chat", "Data"])
with ask_tab:
    history = st.session_state.setdefault("chat_history", [])
    pending_memory = st.session_state.get("pending_memory")
    if pending_memory:
        st.info(f"Save this as a reusable definition?  \n{pending_memory}")
        with st.form("confirm_chat_memory"):
            memory_scope = st.checkbox(
                "Apply across datasets in this workspace", value=False
            )
            left, right = st.columns(2)
            confirm_memory = left.form_submit_button("Save definition", type="primary")
            cancel_memory = right.form_submit_button("Cancel")
        if confirm_memory or cancel_memory:
            if confirm_memory:
                try:
                    memory_item = parse_plain_definition(pending_memory)
                    item = store().upsert(
                        memory_item,
                        None if memory_scope else info["fingerprint"],
                    )
                    st.success(
                        f"Definition {'updated' if item['updated'] else 'saved'}: {item['key']}."
                    )
                except (ValueError, OSError, sqlite3.Error) as exc:
                    st.error(f"Could not save definition: {exc}")
            st.session_state.pop("pending_memory", None)
            st.rerun()
    render_quick_tools(df, info)
    if not history:
        st.caption(
            "Start with a question about a column, a comparison, missing values, or a trend."
        )
        suggestions = suggestions_for(types, info.get("roles", {}))
        for row_start in range(0, len(suggestions), 3):
            row_suggestions = suggestions[row_start : row_start + 3]
            qcols = st.columns(len(row_suggestions))
            for offset, suggestion in enumerate(row_suggestions):
                ix = row_start + offset
                if qcols[offset].button(
                    suggestion, key=f"starter_{ix}", width="stretch"
                ):
                    st.session_state.pending_question = suggestion

    for entry in history:
        st.markdown(
            '<div class="user-row"><div class="user-bubble">'
            f"{html.escape(entry['question'])}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(entry["result"]["answer"])
        if entry["result"].get("memory_subjects"):
            st.caption("Used memory: " + ", ".join(entry["result"]["memory_subjects"]))
        tool_count = len(entry["result"].get("trace", []))
        if not tool_count:
            st.caption("No analysis tool was needed for this answer.")
        else:
            for item in entry["result"].get("trace", []):
                render_evidence(item)
        with st.expander(
            f"Steps · {tool_count} tool call{'s' if tool_count != 1 else ''}"
        ):
            if not tool_count:
                st.caption("No tool calls were made.")
            for item in entry["result"].get("trace", []):
                st.code(
                    json.dumps(item, ensure_ascii=False, indent=2, default=str),
                    language="json",
                )

    prompt = st.chat_input("Ask a question about this dataset")
    prompt = st.session_state.pop("pending_question", None) or prompt
    if prompt:
        clean_prompt = prompt.strip()
        memory_sentence = (
            clean_prompt.split(":", 1)[1].strip()
            if clean_prompt.casefold().startswith("remember:")
            else clean_prompt
        )
        is_memory_intent = clean_prompt.casefold().startswith("remember:") or (
            not clean_prompt.endswith("?")
            and bool(
                re.search(
                    r"\b(means|excludes|includes|defined as|always use)\b",
                    clean_prompt,
                    re.IGNORECASE,
                )
            )
        )
        if is_memory_intent:
            st.session_state.pending_memory = memory_sentence
            st.rerun()
        try:
            active_memory = store().relevant(prompt, info["fingerprint"])
            with st.status("Analyzing the table…", expanded=True) as progress:
                result = ask(
                    prompt,
                    df,
                    types,
                    model=selected_model,
                    max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
                    memories=active_memory,
                    roles=info.get("roles", {}),
                    conversions=info.get("conversions", {}),
                    on_tool=lambda name, args: progress.update(
                        label=running_label(name, args)
                    ),
                )
                progress.update(
                    label="Analysis complete", state="complete", expanded=False
                )
            st.session_state.chat_history.append({"question": prompt, "result": result})
            st.session_state.last_result = result
            st.session_state.last_question = prompt
            st.rerun()
        except (AgentError, ValueError, OSError, sqlite3.Error) as exc:
            st.error(str(exc))

with data_tab:
    st.markdown("### Preview")
    preview_rows = min(50, len(df))
    st.caption(f"First {preview_rows:,} of {len(df):,} rows")
    st.dataframe(
        df.head(preview_rows),
        width="stretch",
        hide_index=True,
        height=min(430, 42 + preview_rows * 35),
    )

    st.markdown("### Columns")
    filter_choice = st.segmented_control(
        "Filter columns",
        ["All", "With missing", "Numeric", "Text"],
        default="All",
        label_visibility="collapsed",
        key="column_filter",
    )
    column_details = pd.DataFrame(prof["column_details"])
    if filter_choice == "With missing":
        column_details = column_details[column_details["missing"] > 0]
    elif filter_choice == "Numeric":
        column_details = column_details[column_details["role"] == "numeric_measure"]
    elif filter_choice == "Text":
        column_details = column_details[
            column_details["role"].isin(
                ["category", "multi_value_category", "long_text"]
            )
        ]
    column_details = column_details.rename(
        columns={
            "name": "Column",
            "type": "Type",
            "role": "Role",
            "missing": "Missing",
            "missing_pct": "Missing %",
            "distinct": "Distinct",
            "examples": "Examples",
            "conversion": "Conversion note",
        }
    )
    st.dataframe(column_details, width="stretch", hide_index=True)
    st.caption(f"Showing {len(column_details):,} matching columns")
