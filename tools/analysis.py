"""Bounded analysis functions for the active dataset."""

from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy import stats


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str = Field(description="Column used to evaluate this filter condition.")
    operator: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "is_null", "not_null"
    ] = Field(
        description="Comparison operation; is_null and not_null do not require a value."
    )
    value: str | int | float | bool | list[str] | None = Field(
        default=None, description="Value or list of values to compare with the column."
    )


class DescribeArgs(Args):
    columns: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Optional subset of columns to profile; omit to include every column.",
    )


class FilterArgs(Args):
    conditions: list[Condition] = Field(
        min_length=1, max_length=8, description="One to eight row filters."
    )
    join: Literal["and", "or"] = Field(
        default="and", description="Require every filter or any filter to match."
    )
    sort_by: str | None = Field(
        default=None, description="Optional result sort column."
    )
    ascending: bool = Field(default=True, description="Sort ascending when true.")
    return_columns: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Optional columns to include in returned rows.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum rows returned; the total match count is still reported.",
    )
    count_only: bool = Field(
        default=False, description="Return the match count without row samples."
    )


class GroupArgs(Args):
    group_by: list[str] = Field(
        min_length=1, max_length=3, description="One to three grouping columns."
    )
    metric: str | None = Field(
        default=None,
        description="Numeric measure for an aggregation; count can count rows without a metric.",
    )
    agg: Literal["mean", "median", "sum", "count", "min", "max", "nunique"] = Field(
        default="count", description="Aggregation applied within each group."
    )
    top_n: int = Field(
        default=20, ge=1, le=100, description="Maximum number of groups returned."
    )
    descending: bool = Field(
        default=True, description="Sort aggregate values highest first when true."
    )
    explode: bool = Field(
        default=False,
        description="Split comma/semicolon/pipe multi-value categories into separate group memberships.",
    )
    filters: list[Condition] | None = Field(
        default=None,
        max_length=5,
        description="Optional pre-aggregation filters; all must match.",
    )


class CorrelationArgs(Args):
    columns: list[str] | None = Field(
        default=None, max_length=20, description="Numeric measures to compare pairwise."
    )
    target: str | None = Field(
        default=None,
        description="Optional target measure to compare with each listed measure.",
    )
    method: Literal["pearson", "spearman"] = Field(
        default="pearson", description="Correlation method; Spearman uses ranks."
    )


class TestArgs(Args):
    test: Literal["welch_t", "anova", "chi_square"] = Field(
        description="Test to run; Welch compares two means, ANOVA compares three or more, chi_square tests categorical association."
    )
    numeric_column: str | None = Field(
        default=None, description="Numeric measure for Welch t-test or ANOVA."
    )
    group_column: str | None = Field(
        default=None,
        description="Category identifying groups for Welch t-test or ANOVA.",
    )
    group_a: str | None = Field(
        default=None, description="First category value for Welch t-test."
    )
    group_b: str | None = Field(
        default=None, description="Second category value for Welch t-test."
    )
    category_column: str | None = Field(
        default=None, description="First categorical variable for chi-square."
    )
    category_column_2: str | None = Field(
        default=None, description="Second categorical variable for chi-square."
    )


class PlotArgs(Args):
    chart: Literal["histogram", "bar", "scatter", "line", "box"] = Field(
        description="Chart form to render."
    )
    x_column: str = Field(
        description="Category/date/measure column for the horizontal axis or distribution."
    )
    y_column: str | None = Field(
        default=None,
        description="Numeric measure for scatter/line charts or optional bar aggregation.",
    )
    aggregation: Literal["mean", "sum", "count", "median"] = Field(
        default="mean",
        description="Aggregation used for bars when y_column is supplied.",
    )


class OutlierArgs(Args):
    column: str = Field(description="Numeric measure to check for unusual values.")
    method: Literal["iqr", "zscore"] = Field(
        default="iqr", description="IQR fences or absolute z-score."
    )
    threshold: float = Field(
        default=3,
        gt=0,
        le=10,
        description="Absolute z-score cutoff; used only by zscore.",
    )
    id_columns: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Optional identifying columns to include with flagged values.",
    )


class MissingArgs(Args):
    top_n: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum columns to list, ordered by missing count.",
    )


class TrendArgs(Args):
    time_column: str = Field(description="Date or year column used to form periods.")
    metric: str | None = Field(
        default=None,
        description="Optional numeric measure to aggregate; omit to count rows.",
    )
    frequency: Literal["day", "week", "month", "quarter", "year", "decade"] = Field(
        default="year", description="Period size for the trend."
    )
    agg: Literal["count", "mean", "sum", "median"] = Field(
        default="count", description="Aggregation for metric values."
    )


class RankArgs(Args):
    sort_by: str = Field(
        description="Measure used to rank rows, or the default measure for group ranking."
    )
    top_n: int = Field(
        default=10, ge=1, le=100, description="Number of top or bottom results."
    )
    group_by: list[str] | None = Field(
        default=None,
        max_length=3,
        description="Optional columns for ranking aggregate groups instead of individual rows.",
    )
    metric: str | None = Field(
        default=None,
        description="Numeric measure aggregated when group_by is provided.",
    )
    agg: Literal["mean", "median", "sum", "count", "min", "max"] = Field(
        default="mean", description="Aggregation used for grouped ranking."
    )
    min_count: int = Field(
        default=1, ge=1, description="Minimum number of records in each ranked group."
    )
    display_columns: list[str] | None = Field(
        default=None, max_length=20, description="Columns to return for row ranking."
    )
    descending: bool = Field(
        default=True,
        description="Return highest values first when true; lowest first otherwise.",
    )


TOOL_SCHEMAS: dict[str, tuple[type[BaseModel], str]] = {
    "describe_dataset": (
        DescribeArgs,
        'Profile columns, types, semantic roles, missing rates, distinct counts, examples, and numeric summary. Use this first to understand a table or inspect selected columns. Arguments: columns is an optional list of column names. Example: {"columns":["Gross","Genre"]}.',
    ),
    "filter_rows": (
        FilterArgs,
        'Find actual records using one or more conditions. Use AND/OR, optional sort and selected output columns; count_only avoids returning rows. Operators: eq, ne, gt, gte, lt, lte, contains, in, is_null, not_null. Example: {"conditions":[{"column":"IMDB_Rating","operator":"gte","value":8.5}],"count_only":true}.',
    ),
    "group_aggregate": (
        GroupArgs,
        'Aggregate a numeric measure or count rows by one or more columns. Use explode=true for comma/semicolon/pipe separated multi-value categories such as Genre. Arguments: group_by, optional metric, agg, top_n, descending, explode, optional filters. Example: {"group_by":["Director"],"metric":"IMDB_Rating","agg":"mean","filters":[{"column":"Film_Count","operator":"gte","value":3}]}.',
    ),
    "correlation": (
        CorrelationArgs,
        'Compute pairwise Pearson or Spearman correlation for numeric measures, optionally against one target. Returns pair counts and strongest results. Arguments: columns or target, method. Example: {"columns":["IMDB_Rating","Meta_score","Gross"],"method":"spearman"}.',
    ),
    "statistical_test": (
        TestArgs,
        'Test a numeric measure across two groups (Welch t-test) or three or more groups (one-way ANOVA), or test association between two categories (chi-square). Report sample sizes and effect size; this does not prove causation. Example: {"test":"welch_t","numeric_column":"IMDB_Rating","group_column":"Genre","group_a":"Drama","group_b":"Comedy"}.',
    ),
    "plot_chart": (
        PlotArgs,
        'Create a PNG histogram, bar chart, scatter plot, line plot, or box plot and return its file path. Choose numeric/date columns that fit the chart. Bars are limited to the top 15 values and scatter plots sample at most 1,000 rows. Example: {"chart":"histogram","x_column":"IMDB_Rating"}.',
    ),
    "detect_outliers": (
        OutlierArgs,
        'Flag unusual values in a numeric measure using IQR fences or absolute z-score. Returns thresholds, count, percentage and a small row sample. Example: {"column":"Gross","method":"iqr","id_columns":["Series_Title"]}.',
    ),
    "analyze_missingness": (
        MissingArgs,
        'Report missing rate per column, rows with any missing value, and columns often missing together. Example: {"top_n":20}.',
    ),
    "trend_over_time": (
        TrendArgs,
        'Summarize a metric or row count over date or year columns by day, week, month, quarter, year or decade. Year columns can be numeric or text. Example: {"time_column":"Released_Year","frequency":"decade","agg":"count"}.',
    ),
    "top_n": (
        RankArgs,
        'Rank rows by a measure, or aggregate and rank groups with an optional minimum group size. Arguments: sort_by, top_n, optional group_by and metric, agg, min_count, display_columns, descending. Example: {"sort_by":"IMDB_Rating","group_by":["Director"],"metric":"IMDB_Rating","agg":"mean","min_count":3,"top_n":10}.',
    ),
}

_OPS = {"eq": "eq", "ne": "ne", "gt": "gt", "gte": "ge", "lt": "lt", "lte": "le"}


def _normal(value: str) -> str:
    return re.sub(r"[\W_]", "", value.casefold())


def _resolve(frame: pd.DataFrame, requested: str) -> str:
    columns = list(map(str, frame.columns))
    if requested in columns:
        return requested
    key = _normal(requested)
    exact = [name for name in columns if _normal(name) == key]
    if len(exact) == 1:
        return exact[0]
    scored = sorted(
        ((SequenceMatcher(None, key, _normal(name)).ratio(), name) for name in columns),
        reverse=True,
    )
    if (
        scored
        and scored[0][0] >= 0.84
        and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.08)
    ):
        return scored[0][1]
    suggestions = [name for score, name in scored if score >= 0.65][:3]
    hint = f" Did you mean {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(
        f"Unknown column '{requested}'.{hint} Available columns: {', '.join(columns)}"
    )


def _resolve_args(frame: pd.DataFrame, value):
    if isinstance(value, dict):
        return {
            k: _resolve_args(frame, v)
            if k in {"conditions", "filters"}
            else _resolve(frame, v)
            if k
            in {
                "column",
                "numeric_column",
                "group_column",
                "category_column",
                "category_column_2",
                "metric",
                "sort_by",
                "time_column",
                "x_column",
                "y_column",
                "target",
            }
            and isinstance(v, str)
            else [_resolve(frame, x) for x in v]
            if k
            in {
                "columns",
                "group_by",
                "display_columns",
                "return_columns",
                "id_columns",
            }
            and isinstance(v, list)
            else _resolve_args(frame, v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_resolve_args(frame, item) for item in value]
    return value


def _numeric(frame: pd.DataFrame, column: str, roles: dict) -> pd.Series:
    role = roles.get(
        column,
        "numeric_measure"
        if pd.api.types.is_numeric_dtype(frame[column])
        else "category",
    )
    if role != "numeric_measure" or not pd.api.types.is_numeric_dtype(frame[column]):
        raise ValueError(
            f"'{column}' is not a numeric measure. Choose a column marked numeric_measure in the profile."
        )
    return pd.to_numeric(frame[column], errors="coerce")


def _mask(frame: pd.DataFrame, condition: dict) -> pd.Series:
    column = condition["column"]
    series = frame[column]
    op, value = condition["operator"], condition.get("value")
    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    if op == "contains":
        return series.astype("string").str.contains(
            re.escape(str(value)), case=False, na=False
        )
    if op == "in":
        if not isinstance(value, list):
            raise ValueError("The 'in' operator needs a list value.")
        return series.isin(value)
    if pd.api.types.is_numeric_dtype(series):
        value = float(value)
    elif pd.api.types.is_datetime64_any_dtype(series):
        value = pd.to_datetime(value, utc=True)
    try:
        return getattr(series, _OPS[op])(value).fillna(False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Condition does not fit column '{column}': {exc}") from exc


def _conditions(
    frame: pd.DataFrame, conditions: list[dict] | None, join: str = "and"
) -> pd.DataFrame:
    if not conditions:
        return frame
    masks = [_mask(frame, condition) for condition in conditions]
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined & mask if join == "and" else combined | mask
    return frame.loc[combined]


def _py(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _base(result: dict) -> dict:
    return {"ok": True, "evidence_id": "ev-" + uuid.uuid4().hex[:10], **result}


def _describe(
    frame: pd.DataFrame,
    types: dict,
    roles: dict,
    conversions: dict,
    columns: list[str] | None,
) -> dict:
    selected = columns or list(frame.columns)
    details = []
    for column in selected[:50]:
        values = frame[column].dropna()
        row = {
            "name": column,
            "dtype": str(frame[column].dtype),
            "role": roles.get(column, types.get(column, "category")),
            "missing": int(frame[column].isna().sum()),
            "missing_pct": round(100 * frame[column].isna().mean(), 2),
            "distinct": int(frame[column].nunique(dropna=True)),
            "examples": [str(v)[:80] for v in values.head(3).tolist()],
            "conversion": conversions.get(column),
        }
        if roles.get(column) == "numeric_measure" and pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            numeric = frame[column].dropna()
            if len(numeric):
                row["summary"] = {
                    "mean": _py(numeric.mean()),
                    "median": _py(numeric.median()),
                    "min": _py(numeric.min()),
                    "max": _py(numeric.max()),
                }
        details.append(row)
    return {"rows": len(frame), "columns": len(frame.columns), "profile": details}


def _filter_rows(frame: pd.DataFrame, a: dict) -> dict:
    selected = _conditions(
        frame,
        [c.model_dump() if isinstance(c, BaseModel) else c for c in a["conditions"]],
        a["join"],
    )
    matched = len(selected)
    if a["count_only"]:
        return {
            "matched_rows": matched,
            "rows": [],
            "returned_rows": 0,
            "count_only": True,
        }
    if a["sort_by"]:
        selected = selected.sort_values(
            a["sort_by"], ascending=a["ascending"], na_position="last"
        )
    columns = a["return_columns"] or list(frame.columns[:20])
    result = selected.loc[:, columns].head(a["limit"])
    return {
        "matched_rows": matched,
        "returned_rows": len(result),
        "count_only": False,
        "rows": result.astype(object).where(pd.notna(result), None).to_dict("records"),
    }


def _aggregate(frame: pd.DataFrame, a: dict, roles: dict) -> dict:
    filtered = _conditions(
        frame,
        [
            c.model_dump() if isinstance(c, BaseModel) else c
            for c in (a.get("filters") or [])
        ],
    )
    data = filtered.copy()
    keys = a["group_by"]
    if a["explode"]:
        for key in keys:
            if roles.get(key) == "multi_value_category":
                data[key] = data[key].astype("string").str.split(r"\s*[,;|]\s*")
                data = data.explode(key)
    metric = a.get("metric")
    operation = a["agg"]
    if operation != "count" and not metric:
        raise ValueError(f"'{operation}' requires a numeric metric column.")
    if metric and operation in {"mean", "median", "sum", "min", "max"}:
        data[metric] = _numeric(data, metric, roles)
    if metric:
        result = (
            data.groupby(keys, dropna=False)[metric]
            .agg(operation)
            .rename("value")
            .reset_index()
        )
    else:
        result = data.groupby(keys, dropna=False).size().rename("value").reset_index()
    result = result.sort_values(
        "value", ascending=not a["descending"], na_position="last"
    ).head(a["top_n"])
    result["value"] = result["value"].map(_py)
    return {
        "group_by": keys,
        "metric": metric,
        "aggregation": operation,
        "groups": result.to_dict("records"),
        "rows_considered": len(filtered),
    }


def _correlation(frame: pd.DataFrame, a: dict, roles: dict) -> dict:
    candidates = a["columns"] or [
        c for c, role in roles.items() if role == "numeric_measure"
    ]
    if a["target"]:
        candidates = [a["target"]] + [c for c in candidates if c != a["target"]]
    if len(candidates) < 2:
        raise ValueError("Choose at least two numeric_measure columns.")
    if len(candidates) > 20:
        candidates = candidates[:20]
    values = pd.DataFrame({c: _numeric(frame, c, roles) for c in candidates})
    pairs = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            subset = values[[left, right]].dropna()
            ok = (
                len(subset) >= 2
                and subset[left].nunique() > 1
                and subset[right].nunique() > 1
            )
            coefficient = (
                subset[left].corr(subset[right], method=a["method"]) if ok else None
            )
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "coefficient": _py(coefficient),
                    "n": len(subset),
                    "method": a["method"],
                }
            )
    pairs.sort(key=lambda p: abs(p["coefficient"] or 0), reverse=True)
    return {
        "pairs": pairs[:30],
        "note": "Correlation is an association, not evidence of causation.",
    }


def _stat_test(frame: pd.DataFrame, a: dict, roles: dict) -> dict:
    kind = a["test"]
    if kind in {"welch_t", "anova"}:
        numeric_column, group_column = a["numeric_column"], a["group_column"]
        values = _numeric(frame, numeric_column, roles)
        data = pd.DataFrame({"value": values, "group": frame[group_column]}).dropna()
        if roles.get(group_column) == "multi_value_category":
            data["group"] = data["group"].astype("string").str.split(r"\s*[,;|]\s*")
            data = data.explode("group")
        groups = {
            str(name): group["value"].to_numpy()
            for name, group in data.groupby("group")
        }
        if kind == "welch_t":
            if not a["group_a"] or not a["group_b"]:
                raise ValueError("Welch t-test requires group_a and group_b.")
            first, second = (
                groups.get(str(a["group_a"]), np.array([])),
                groups.get(str(a["group_b"]), np.array([])),
            )
            if min(len(first), len(second)) < 2:
                raise ValueError(
                    "Each selected group needs at least two numeric observations."
                )
            result = stats.ttest_ind(first, second, equal_var=False)
            pooled = np.sqrt(
                (
                    (len(first) - 1) * np.var(first, ddof=1)
                    + (len(second) - 1) * np.var(second, ddof=1)
                )
                / (len(first) + len(second) - 2)
            )
            effect = (
                float((np.mean(first) - np.mean(second)) / pooled) if pooled else None
            )
            return {
                "test": kind,
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "degrees_of_freedom": float(result.df),
                "group_sizes": {
                    str(a["group_a"]): len(first),
                    str(a["group_b"]): len(second),
                },
                "effect_size": {"name": "cohens_d", "value": effect},
                "reading": "Statistically significant at 0.05."
                if result.pvalue < 0.05
                else "No statistically significant difference at 0.05.",
                "assumption_note": "Welch's test compares means and does not assume equal variances.",
            }
        arrays = [values for values in groups.values() if len(values) >= 2]
        if len(arrays) < 3:
            raise ValueError(
                "ANOVA needs at least three groups with two observations each."
            )
        result = stats.f_oneway(*arrays)
        grand = data["value"].mean()
        ss_total = ((data["value"] - grand) ** 2).sum()
        ss_between = sum(len(group) * (group.mean() - grand) ** 2 for group in arrays)
        return {
            "test": kind,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "degrees_of_freedom": [len(arrays) - 1, len(data) - len(arrays)],
            "group_sizes": {k: len(v) for k, v in groups.items()},
            "effect_size": {
                "name": "eta_squared",
                "value": float(ss_between / ss_total) if ss_total else None,
            },
            "reading": "At least one group mean differs at 0.05."
            if result.pvalue < 0.05
            else "No group mean difference detected at 0.05.",
            "assumption_note": "One-way ANOVA assumes independent observations and approximately normal within-group residuals.",
        }
    first, second = a["category_column"], a["category_column_2"]
    table = pd.crosstab(frame[first], frame[second])
    if min(table.shape) < 2:
        raise ValueError(
            "Chi-square needs at least two observed categories in each column."
        )
    statistic, p_value, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = (
        np.sqrt(statistic / (n * min(table.shape[0] - 1, table.shape[1] - 1)))
        if n
        else None
    )
    small = bool((expected < 5).any())
    return {
        "test": kind,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "group_sizes": {"rows": int(n), "table_shape": list(table.shape)},
        "effect_size": {
            "name": "cramers_v",
            "value": float(v) if v is not None else None,
        },
        "reading": "The categories are associated at 0.05."
        if p_value < 0.05
        else "No category association detected at 0.05.",
        "assumption_note": "Some expected counts are below 5; interpret the chi-square approximation with care."
        if small
        else "Chi-square assumes independent records and adequate expected cell counts.",
    }


def _plot(frame: pd.DataFrame, a: dict, roles: dict) -> dict:
    chart, x, y = a["chart"], a["x_column"], a["y_column"]
    if chart in {"histogram", "box"}:
        values = _numeric(frame, x, roles).dropna()
    elif chart in {"scatter", "line"}:
        if not y:
            raise ValueError(f"{chart} requires y_column.")
        if roles.get(x) == "date":
            values_x = pd.to_datetime(frame[x], errors="coerce")
        elif roles.get(x) == "year":
            year_text = (
                pd.to_numeric(frame[x], errors="coerce")
                .astype("Int64")
                .astype("string")
            )
            values_x = pd.to_datetime(year_text, format="%Y", errors="coerce")
        else:
            values_x = _numeric(frame, x, roles)
        values_y = _numeric(frame, y, roles)
        points = pd.DataFrame({"x": values_x, "y": values_y}).dropna()
    elif chart == "bar":
        categories = frame[x].astype("string").fillna("<missing>")
        if y:
            measure = _numeric(frame, y, roles)
            grouped = (
                pd.DataFrame({"category": categories, "value": measure})
                .groupby("category")["value"]
                .agg(a["aggregation"])
                .sort_values(ascending=False)
            )
        else:
            grouped = categories.value_counts()
        if len(grouped) > 15:
            top = grouped.head(15)
            grouped = pd.concat([top, pd.Series({"Other": grouped.iloc[15:].sum()})])
    output = Path(__file__).parents[1] / "outputs"
    output.mkdir(exist_ok=True)
    path = output / f"chart-{uuid.uuid4().hex[:10]}.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if chart == "histogram":
        if not len(values):
            raise ValueError("No numeric values are available for this histogram.")
        ax.hist(values, bins="auto", color="#1f5f4a", edgecolor="white")
        ax.set_xlabel(x)
        ax.set_ylabel("Rows")
    elif chart == "box":
        if not len(values):
            raise ValueError("No numeric values are available for this box plot.")
        ax.boxplot(values, vert=False)
        ax.set_xlabel(x)
    elif chart == "bar":
        if not len(grouped):
            raise ValueError("No values are available for this bar chart.")
        grouped.plot.bar(ax=ax, color="#1f5f4a")
        ax.set_xlabel(x)
        ax.tick_params(axis="x", labelrotation=35)
    elif chart in {"scatter", "line"}:
        if len(points) > 1000:
            points = points.sample(1000, random_state=7)
        points = points.sort_values("x")
        if chart == "scatter":
            ax.scatter(points["x"], points["y"], alpha=0.65, color="#1f5f4a")
        else:
            ax.plot(points["x"], points["y"], color="#1f5f4a")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    else:
        plt.close(fig)
        raise ValueError("Unsupported chart type.")
    ax.set_title(
        {
            "histogram": "Distribution",
            "bar": "Values by category",
            "scatter": "Relationship",
            "line": "Values over time",
            "box": "Value spread",
        }[chart]
    )
    fig.savefig(path, format="png", dpi=120)
    plt.close(fig)
    return {
        "chart": chart,
        "file_path": str(path.resolve()),
        "rows_plotted": len(values)
        if chart in {"histogram", "box"}
        else len(grouped)
        if chart == "bar"
        else len(points),
    }


def _outliers(frame: pd.DataFrame, a: dict, roles: dict) -> dict:
    values = _numeric(frame, a["column"], roles)
    valid = values.dropna()
    if len(valid) < 4:
        raise ValueError("Outlier checks need at least four numeric values.")
    if a["method"] == "iqr":
        q1, q3 = valid.quantile([0.25, 0.75])
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
    else:
        mean, std = valid.mean(), valid.std(ddof=0)
        if std == 0:
            raise ValueError("All values are equal; z-scores cannot flag outliers.")
        low, high = mean - a["threshold"] * std, mean + a["threshold"] * std
    mask = values.lt(low) | values.gt(high)
    cols = list(dict.fromkeys([*a["id_columns"], a["column"]]))
    sample = (
        frame.loc[mask, cols]
        .head(20)
        .astype(object)
        .where(pd.notna(frame.loc[mask, cols].head(20)), None)
        .to_dict("records")
    )
    return {
        "column": a["column"],
        "method": a["method"],
        "lower_threshold": _py(low),
        "upper_threshold": _py(high),
        "count": int(mask.sum()),
        "percentage": round(100 * float(mask.mean()), 2),
        "sample_rows": sample,
    }


def _missing(frame: pd.DataFrame, a: dict) -> dict:
    missing = frame.isna()
    counts = missing.sum().sort_values(ascending=False)
    cols = list(counts[counts > 0].head(20).index)
    pairs = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            both = int((missing[left] & missing[right]).sum())
            if both:
                pairs.append({"columns": [left, right], "both_missing": both})
    return {
        "rows_with_missing": int(missing.any(axis=1).sum()),
        "complete_rows": int((~missing.any(axis=1)).sum()),
        "columns": [
            {
                "column": c,
                "missing": int(n),
                "percentage": round(100 * n / len(frame), 2) if len(frame) else 0,
            }
            for c, n in counts.head(a["top_n"]).items()
        ],
        "co_missing_pairs": sorted(
            pairs, key=lambda item: item["both_missing"], reverse=True
        )[:30],
    }


def _trend(frame: pd.DataFrame, a: dict, roles: dict) -> dict:
    column = a["time_column"]
    role = roles.get(column)
    if role not in {"year", "date"}:
        raise ValueError(f"'{column}' must be typed as year or date.")
    if role == "year":
        years = pd.to_numeric(frame[column], errors="coerce")
        if a["frequency"] == "decade":
            labels = (years // 10 * 10).astype("Int64").astype("string") + "s"
        else:
            labels = years.round().astype("Int64").astype("string")
        valid = labels.notna() & labels.ne("<NA>")
        data = pd.DataFrame({"period": labels[valid]})
    else:
        dates = pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_convert(
            None
        )
        if a["frequency"] == "decade":
            labels = (dates.dt.year // 10 * 10).astype("Int64").astype("string") + "s"
        elif a["frequency"] == "year":
            labels = dates.dt.year.astype("Int64").astype("string")
        elif a["frequency"] == "quarter":
            labels = dates.dt.to_period("Q").astype("string")
        elif a["frequency"] == "month":
            labels = dates.dt.to_period("M").astype("string")
        elif a["frequency"] == "week":
            labels = dates.dt.to_period("W").astype("string")
        else:
            labels = dates.dt.date.astype("string")
        valid = dates.notna()
        data = pd.DataFrame({"period": labels[valid]})
    metric = a.get("metric")
    if metric:
        data["metric"] = _numeric(frame.loc[data.index], metric, roles)
        agg = a["agg"] if a["agg"] != "count" else "count"
        grouped = data.groupby("period")["metric"].agg(agg)
    else:
        grouped = data.groupby("period").size()
    grouped = grouped.sort_index()
    result = []
    previous = None
    for period, value in grouped.items():
        number = float(value)
        result.append(
            {
                "period": str(period),
                "value": _py(value),
                "change": _py(number - previous) if previous is not None else None,
            }
        )
        previous = number
    return {
        "time_column": column,
        "metric": metric,
        "frequency": a["frequency"],
        "periods": result[:240],
        "rows_counted": len(data),
    }


def _rank(frame: pd.DataFrame, a: dict, roles: dict) -> dict:
    if a["group_by"]:
        metric = a.get("metric") or a["sort_by"]
        values = _numeric(frame, metric, roles)
        data = frame.assign(_value=values).dropna(subset=["_value"])
        result = (
            data.groupby(a["group_by"], dropna=False)["_value"]
            .agg([a["agg"], "count"])
            .reset_index()
            .rename(columns={a["agg"]: "value"})
        )
        result = (
            result[result["count"] >= a["min_count"]]
            .sort_values("value", ascending=not a["descending"])
            .head(a["top_n"])
        )
        return {
            "ranked_by": f"{a['agg']}({metric})",
            "groups": result.astype(object)
            .where(pd.notna(result), None)
            .to_dict("records"),
            "returned_rows": len(result),
        }
    columns = a["display_columns"] or list(frame.columns[:20])
    result = (
        frame.sort_values(
            a["sort_by"], ascending=not a["descending"], na_position="last"
        )
        .loc[:, columns]
        .head(a["top_n"])
    )
    return {
        "ranked_by": a["sort_by"],
        "rows": result.astype(object).where(pd.notna(result), None).to_dict("records"),
        "returned_rows": len(result),
    }


def _execute(
    name: str,
    frame: pd.DataFrame,
    types: dict,
    roles: dict,
    conversions: dict,
    arguments: dict,
) -> dict:
    if name == "describe_dataset":
        return _describe(frame, types, roles, conversions, arguments.get("columns"))
    if name == "filter_rows":
        return _filter_rows(frame, arguments)
    if name == "group_aggregate":
        return _aggregate(frame, arguments, roles)
    if name == "correlation":
        return _correlation(frame, arguments, roles)
    if name == "statistical_test":
        return _stat_test(frame, arguments, roles)
    if name == "plot_chart":
        return _plot(frame, arguments, roles)
    if name == "detect_outliers":
        return _outliers(frame, arguments, roles)
    if name == "analyze_missingness":
        return _missing(frame, arguments)
    if name == "trend_over_time":
        return _trend(frame, arguments, roles)
    if name == "top_n":
        return _rank(frame, arguments, roles)
    raise ValueError(f"Unknown tool '{name}'.")


def run_tool(
    name: str,
    frame: pd.DataFrame,
    types: dict,
    arguments: dict,
    roles: dict | None = None,
    conversions: dict | None = None,
) -> dict:
    if name not in TOOL_SCHEMAS:
        return {
            "ok": False,
            "evidence_id": "ev-" + uuid.uuid4().hex[:10],
            "error": {
                "type": "UnknownTool",
                "message": f"Unknown tool '{name}'. Choose from: {', '.join(TOOL_SCHEMAS)}.",
            },
        }
    roles = roles or {
        c: (
            "numeric_measure"
            if kind == "numeric"
            else "date"
            if kind == "datetime"
            else kind
        )
        for c, kind in types.items()
    }
    try:
        resolved = _resolve_args(frame, arguments)
        args = TOOL_SCHEMAS[name][0].model_validate(resolved).model_dump()
        result = _execute(name, frame, types, roles, conversions or {}, args)
        return _base(result)
    except (ValidationError, ValueError, TypeError, KeyError, IndexError) as exc:
        return {
            "ok": False,
            "evidence_id": "ev-" + uuid.uuid4().hex[:10],
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def tool_definitions() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": model.model_json_schema(),
            },
        }
        for name, (model, description) in TOOL_SCHEMAS.items()
    ]
