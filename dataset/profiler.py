"""Dataset profile generation."""

import pandas as pd


def profile_dataset(
    frame: pd.DataFrame,
    types: dict[str, str],
    roles: dict[str, str] | None = None,
    conversions: dict[str, str] | None = None,
) -> dict:
    roles = roles or {}
    conversions = conversions or {}
    cols = []
    for c in frame.columns:
        s = frame[c]
        miss = int(s.isna().sum())
        cols.append(
            {
                "name": str(c),
                "type": types.get(c, "unknown"),
                "role": roles.get(c, "category"),
                "missing": miss,
                "missing_pct": round(miss * 100 / len(frame), 2) if len(frame) else 0,
                "distinct": int(s.nunique(dropna=True)),
                "examples": [str(value)[:80] for value in s.dropna().head(3).tolist()],
                "conversion": conversions.get(c),
            }
        )
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "column_details": cols,
    }
