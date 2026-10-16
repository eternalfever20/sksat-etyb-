from pathlib import Path

import pandas as pd
import pytest
from scipy import stats

from tools.analysis import TOOL_SCHEMAS, run_tool


def call(name, frame, types, args, roles=None):
    return run_tool(name, frame, types, args, roles=roles)


@pytest.fixture
def imdb_table():
    frame = pd.DataFrame(
        {
            "Series_Title": ["A", "B", "C", "D", "E", "F"],
            "Released_Year": [1990, 1991, 2001, 2002, 2010, 2011],
            "IMDB_Rating": [7.0, 8.0, 8.5, 9.0, 6.0, 7.5],
            "Meta_score": [70, None, 85, 95, 60, 75],
            "Genre": [
                "Drama, Crime",
                "Drama",
                "Comedy",
                "Comedy, Drama",
                "Crime",
                "Drama, Crime",
            ],
        }
    )
    types = {
        "Series_Title": "text",
        "Released_Year": "numeric",
        "IMDB_Rating": "numeric",
        "Meta_score": "numeric",
        "Genre": "categorical",
    }
    roles = {
        "Series_Title": "identifier",
        "Released_Year": "year",
        "IMDB_Rating": "numeric_measure",
        "Meta_score": "numeric_measure",
        "Genre": "multi_value_category",
    }
    return frame, types, roles


def test_profile_filter_and_group_aggregation(imdb_table):
    frame, types, roles = imdb_table
    profile = call("describe_dataset", frame, types, {}, roles)
    assert profile["ok"] and profile["rows"] == 6
    assert (
        next(col for col in profile["profile"] if col["name"] == "Released_Year")[
            "role"
        ]
        == "year"
    )
    filtered = call(
        "filter_rows",
        frame,
        types,
        {
            "conditions": [{"column": "IMDB_Rating", "operator": "gte", "value": 8}],
            "count_only": True,
        },
        roles,
    )
    assert filtered["matched_rows"] == 3
    grouped = call(
        "group_aggregate",
        frame,
        types,
        {
            "group_by": ["Genre"],
            "metric": "IMDB_Rating",
            "agg": "mean",
            "explode": True,
        },
        roles,
    )
    assert grouped["ok"] and len(grouped["groups"]) == 3
    assert {row["Genre"] for row in grouped["groups"]} == {"Drama", "Crime", "Comedy"}


def test_correlation_statistical_tests_and_known_values(imdb_table):
    frame, types, roles = imdb_table
    corr = call(
        "correlation",
        frame,
        types,
        {"columns": ["IMDB_Rating", "Meta_score"], "method": "pearson"},
        roles,
    )
    expected = frame[["IMDB_Rating", "Meta_score"]].dropna().corr().iloc[0, 1]
    assert corr["ok"] and corr["pairs"][0]["coefficient"] == pytest.approx(expected)
    test = call(
        "statistical_test",
        frame,
        types,
        {
            "test": "welch_t",
            "numeric_column": "IMDB_Rating",
            "group_column": "Genre",
            "group_a": "Drama",
            "group_b": "Comedy",
        },
        roles,
    )
    assert test["ok"] and test["group_sizes"] == {"Drama": 4, "Comedy": 2}
    drama = [7.0, 8.0, 9.0, 7.5]
    comedy = [8.5, 9.0]
    expected_t = stats.ttest_ind(drama, comedy, equal_var=False)
    assert test["p_value"] == pytest.approx(expected_t.pvalue)
    chi = call(
        "statistical_test",
        frame,
        types,
        {
            "test": "chi_square",
            "category_column": "Genre",
            "category_column_2": "Series_Title",
        },
        roles,
    )
    assert chi["ok"] and chi["effect_size"]["name"] == "cramers_v"
    expected_chi = stats.chi2_contingency(
        pd.crosstab(frame["Genre"], frame["Series_Title"])
    )
    assert chi["statistic"] == pytest.approx(expected_chi.statistic)
    assert chi["p_value"] == pytest.approx(expected_chi.pvalue)


def test_plots_missingness_trend_ranking_and_outliers(
    simple_frame, imdb_table, tmp_path, monkeypatch
):
    frame, types = simple_frame
    roles = {
        "value": "numeric_measure",
        "group": "category",
        "day": "date",
        "flag": "boolean",
    }
    monkeypatch.chdir(tmp_path)
    chart = call(
        "plot_chart", frame, types, {"chart": "histogram", "x_column": "value"}, roles
    )
    assert (
        chart["ok"]
        and Path(chart["file_path"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    )
    missing = call("analyze_missingness", frame, types, {}, roles)
    assert missing["rows_with_missing"] == 1
    trend_frame, trend_types, trend_roles = imdb_table
    trend = call(
        "trend_over_time",
        trend_frame,
        trend_types,
        {"time_column": "Released_Year", "frequency": "decade", "agg": "count"},
        trend_roles,
    )
    assert trend["ok"] and len(trend["periods"]) == 3
    ranked = call(
        "top_n",
        trend_frame,
        trend_types,
        {"sort_by": "IMDB_Rating", "top_n": 2},
        trend_roles,
    )
    assert ranked["rows"][0]["Series_Title"] == "D"
    outliers = call(
        "detect_outliers", frame, types, {"column": "value", "method": "iqr"}, roles
    )
    assert outliers["ok"] and outliers["count"] == 1


def test_validation_fuzzy_column_and_year_not_measure(imdb_table):
    frame, types, roles = imdb_table
    fuzzy = call("top_n", frame, types, {"sort_by": "imdb rating", "top_n": 1}, roles)
    assert fuzzy["ok"] and fuzzy["rows"][0]["Series_Title"] == "D"
    bad = call("top_n", frame, types, {"sort_by": "missing_column"}, roles)
    assert not bad["ok"] and "Available columns" in bad["error"]["message"]
    year_mean = call(
        "group_aggregate",
        frame,
        types,
        {"group_by": ["Genre"], "metric": "Released_Year", "agg": "mean"},
        roles,
    )
    assert not year_mean["ok"] and "numeric measure" in year_mean["error"]["message"]
    invalid = call(
        "filter_rows",
        frame,
        types,
        {"conditions": [{"column": "Genre", "operator": "gt", "value": 1}]},
        roles,
    )
    assert not invalid["ok"]


def test_all_ten_tools_have_strict_schemas():
    assert len(TOOL_SCHEMAS) == 10
    assert {
        "describe_dataset",
        "filter_rows",
        "group_aggregate",
        "correlation",
        "statistical_test",
        "plot_chart",
        "detect_outliers",
        "analyze_missingness",
        "trend_over_time",
        "top_n",
    } == set(TOOL_SCHEMAS)
    for schema, _description in TOOL_SCHEMAS.values():
        assert schema.model_config["extra"] == "forbid"
