import io
from pathlib import Path

import pandas as pd
import pytest

from dataset.loader import DatasetError, load_dataset


def test_demo_csv_is_synthetic_and_profiles_expected_values(demo):
    frame, info = demo
    assert len(frame) == 35
    assert len(frame.columns) == 12
    assert info["duplicate_rows"] == 0
    assert frame["revenue"].isna().sum() == 1
    assert info["types"]["order_date"] == "datetime"
    assert info["types"]["revenue"] == "numeric"
    assert info["types"]["returned"] == "boolean"
    assert frame["revenue"].sum() == pytest.approx(3822.95)


def test_bundled_uci_iris_loads_for_real_analysis():
    path = Path(__file__).parents[1] / "data/iris.csv"
    frame, info = load_dataset(path.name, path.read_bytes())
    assert len(frame) == 150
    assert frame["species"].nunique() == 3
    assert info["roles"]["species"] == "category"
    assert frame["sepal_length_cm"].mean() == pytest.approx(5.8433333333)


def test_csv_encodings_and_duplicate_headers():
    content = "name,name,amount\nJos\u00e9,J,4\n".encode("cp1252")
    frame, _ = load_dataset("sample.csv", content)
    assert list(frame.columns) == ["name", "name_2", "amount"]
    assert frame.loc[0, "name"] == "Jos\u00e9"


def test_empty_unsupported_and_oversize():
    for args in [("x.csv", b""), ("x.txt", b"x"), ("x.csv", b"a,b\n1,2", 0)]:
        with pytest.raises(DatasetError):
            load_dataset(*args)
    with pytest.raises(DatasetError):
        load_dataset("e.csv", b"a,b\n")


def test_excel_multi_sheet_selection():
    buff = io.BytesIO()
    with pd.ExcelWriter(buff, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, index=False, sheet_name="One")
        pd.DataFrame({"x": [2]}).to_excel(writer, index=False, sheet_name="Two")
    with pytest.raises(DatasetError, match="multiple sheets"):
        load_dataset("sample.xlsx", buff.getvalue())
    frame, _ = load_dataset("sample.xlsx", buff.getvalue(), sheet_name="Two")
    assert frame.x.tolist() == [2]


def test_mixed_and_empty_type_inference():
    _, info = load_dataset("x.csv", b"a,b,c\n2025-01-01,true,\n2025-01-02,false,\n")
    assert info["types"] == {"a": "datetime", "b": "boolean", "c": "empty"}


def test_currency_numbers_and_safe_mixed_date_formats():
    content = (
        "Net Amount,Invoice Date,Ambiguous Date,Account ID\n"
        '"$1,200.50",2025-01-03,01/02/2025,0012\n'
        '"(€40.00)",01/15/2025,03/04/2025,0013\n'
        '"£250.00",Feb 03 2025,05/06/2025,0014\n'
    ).encode()
    frame, info = load_dataset("mixed.csv", content)

    assert info["types"]["Net Amount"] == "numeric"
    assert frame["Net Amount"].tolist() == pytest.approx([1200.5, -40, 250])
    assert info["types"]["Invoice Date"] == "datetime"
    assert frame["Invoice Date"].notna().all()
    assert info["types"]["Ambiguous Date"] != "datetime"
    assert info["types"]["Account ID"] != "numeric"


def test_imdb_shaped_semantic_roles_and_conversion_notes():
    rows = [
        (
            f"title-{i}",
            "8.7",
            "1920",
            "142 min",
            "Drama, Crime",
            "PG-13",
            "28,341,469",
            "A long description " + "x" * 120,
        )
        for i in range(40)
    ]
    rows[3] = (*rows[3][:2], "PG", *rows[3][3:])
    rows[4] = (*rows[4][:6], "", rows[4][7])
    body = "Series_Title,IMDB_Rating,Released_Year,Runtime,Genre,Certificate,Gross,Overview\n"
    body += "\n".join(",".join(f'"{v}"' for v in row) for row in rows)
    frame, info = load_dataset("imdb_top_1000.csv", body.encode())

    assert info["roles"]["Series_Title"] == "identifier"
    assert info["roles"]["IMDB_Rating"] == "numeric_measure"
    assert info["roles"]["Released_Year"] == "year"
    assert info["roles"]["Runtime"] == "numeric_measure"
    assert frame["Runtime"].dropna().eq(142).all()
    assert info["roles"]["Genre"] == "multi_value_category"
    assert info["roles"]["Certificate"] == "category"
    assert info["roles"]["Gross"] == "numeric_measure"
    assert frame["Gross"].dropna().eq(28_341_469).all()
    assert info["roles"]["Overview"] == "long_text"
    assert "Parsed year" in info["conversions"]["Released_Year"]
