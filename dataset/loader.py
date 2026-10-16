"""Safe, bounded CSV and Excel ingestion with conservative type inference."""

from __future__ import annotations

import csv
import hashlib
import io
import re
from pathlib import Path

import pandas as pd


class DatasetError(ValueError):
    """A user-correctable dataset loading error."""


_CURRENCY = r"[$€£¥₹]"
_CURRENCY_CODES = r"(?i)(?:USD|EUR|GBP|INR|JPY)"
_UNIT_SUFFIX = r"(?i)(?:min(?:utes?)?|hours?|hrs?|days?|km|mi|kg|g|lb|lbs|miles?)?"
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?$")
_ID_NAME = re.compile(
    r"(?:^|[\W_])(id|identifier|code|zip|postal|phone|account|sku)(?:$|[\W_])",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ].*)?$")
_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:[T ].*)?$")


def _numeric_strings(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace("−", "-", regex=False)
    negative_parentheses = text.str.fullmatch(r"\(.*\)", na=False)
    text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    text = text.str.replace(
        rf"^{_CURRENCY_CODES}\s*|\s*{_CURRENCY_CODES}$", "", regex=True
    )
    text = text.str.replace(rf"^{_CURRENCY}\s*|\s*{_CURRENCY}$", "", regex=True)
    text = text.str.replace(r"%$", "", regex=True)
    text = text.str.replace(rf"\s*{_UNIT_SUFFIX}$", "", regex=True)
    text = text.str.replace(r"\s+", "", regex=True)
    valid = text.str.fullmatch(_NUMBER_PATTERN, na=False)
    parsed = pd.to_numeric(
        text.where(valid).str.replace(",", "", regex=False), errors="coerce"
    )
    return parsed.where(~negative_parentheses, -parsed.abs())


def _safe_dates(series: pd.Series) -> pd.Series | None:
    nonempty = series.dropna().astype("string").str.strip()
    if nonempty.empty:
        return None
    samples = nonempty.tolist()
    has_iso = any(_ISO_DATE.fullmatch(value) for value in samples)
    numeric_parts = [_NUMERIC_DATE.fullmatch(value) for value in samples]
    numeric_parts = [match for match in numeric_parts if match]
    day_first = False
    if numeric_parts:
        first_over_12 = any(int(match.group(1)) > 12 for match in numeric_parts)
        second_over_12 = any(int(match.group(2)) > 12 for match in numeric_parts)
        if first_over_12 and second_over_12:
            return None
        if first_over_12:
            day_first = True
        elif second_over_12:
            day_first = False
        elif not has_iso and not any(
            re.search(r"[A-Za-z]", value) for value in samples
        ):
            # 03/04/2025 is valid in two different orders; do not guess.
            return None
    parsed = pd.to_datetime(
        series, errors="coerce", format="mixed", utc=True, dayfirst=day_first
    )
    if int(parsed.notna().sum()) / len(nonempty) >= 0.9:
        return parsed
    return None


def _dedupe_columns(columns: list[object]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for raw in columns:
        base = str(raw).strip() or "unnamed"
        n = seen.get(base, 0)
        seen[base] = n + 1
        result.append(base if n == 0 else f"{base}_{n + 1}")
    return result


def _infer_types(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, dict]:
    out = frame.copy()
    types: dict[str, str] = {}
    roles: dict[str, str] = {}
    conversions: dict[str, str] = {}
    for col in out.columns:
        s = out[col]
        nonnull = s.dropna()
        name = str(col)
        lower_name = name.casefold()
        is_year = bool(re.search(r"(^|[_\s])year($|[_\s])", lower_name))
        if nonnull.empty:
            types[col] = "empty"
            roles[col] = "category"
        elif pd.api.types.is_bool_dtype(s.dtype):
            types[col] = "boolean"
            roles[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(s.dtype):
            if is_year:
                types[col] = "numeric"
                roles[col] = "year"
            elif _ID_NAME.search(name):
                out[col] = s.astype("string")
                types[col] = "text"
                roles[col] = "identifier"
            else:
                types[col] = "numeric"
                roles[col] = "numeric_measure"
        elif pd.api.types.is_datetime64_any_dtype(s.dtype):
            types[col] = "datetime"
            roles[col] = "date"
        else:
            text = nonnull.astype("string").str.strip()
            lower = set(text.str.lower().unique())
            if lower and lower <= {"true", "false", "yes", "no", "0", "1"}:
                out[col] = (
                    s.astype("string")
                    .str.strip()
                    .str.lower()
                    .map(
                        {
                            "true": True,
                            "yes": True,
                            "1": True,
                            "false": False,
                            "no": False,
                            "0": False,
                        }
                    )
                    .astype("boolean")
                )
                types[col] = "boolean"
                roles[col] = "boolean"
                continue
            year_values = _numeric_strings(s) if is_year else pd.Series(dtype="float64")
            if is_year and year_values.notna().sum() / len(nonnull) >= 0.9:
                out[col] = year_values
                types[col] = "numeric"
                roles[col] = "year"
                conversions[col] = (
                    "Parsed year strings as numbers; non-year values remain missing."
                )
                continue
            if not _ID_NAME.search(name):
                numbers = _numeric_strings(s)
                if numbers.notna().sum() / len(nonnull) >= 0.9:
                    out[col] = numbers
                    types[col] = "numeric"
                    roles[col] = "numeric_measure"
                    conversions[col] = (
                        "Parsed numeric strings and removed supported separators, currency, percent, or unit suffixes."
                    )
                    continue
            dates = _safe_dates(s)
            if dates is not None:
                out[col] = dates
                types[col] = "datetime"
                roles[col] = "date"
                continue
            delimiters = text.str.contains(r"[,;|]", regex=True).mean()
            if delimiters >= 0.25:
                types[col] = "categorical"
                roles[col] = "multi_value_category"
            elif _ID_NAME.search(name) or lower_name in {
                "series_title",
                "movie_title",
                "film_title",
            }:
                types[col] = "text"
                roles[col] = "identifier"
            elif (
                any(
                    token in lower_name
                    for token in (
                        "overview",
                        "description",
                        "plot",
                        "poster",
                        "url",
                        "link",
                    )
                )
                or text.str.len().mean() > 100
            ):
                types[col] = "text"
                roles[col] = "long_text"
            elif len(lower) <= max(30, int(len(nonnull) * 0.2)):
                types[col] = "categorical"
                roles[col] = "category"
            else:
                types[col] = "text"
                roles[col] = "long_text"
    return out, types, roles, conversions


def load_dataset(
    name: str, content: bytes, max_upload_mb: int = 25, sheet_name: str | None = None
) -> tuple[pd.DataFrame, dict]:
    """Load a CSV/XLSX byte stream, enforcing size and structure limits."""
    safe_name = Path(name.replace("\\", "/")).name
    ext = Path(safe_name).suffix.lower()
    if not 1 <= max_upload_mb <= 25:
        raise DatasetError("MAX_UPLOAD_MB must be an integer from 1 through 25.")
    if ext not in {".csv", ".xlsx", ".xlsm"}:
        raise DatasetError("Supported file types are CSV, XLSX, and XLSM.")
    if not content:
        raise DatasetError("The uploaded file is empty.")
    limit = max_upload_mb * 1024 * 1024
    if len(content) > limit:
        raise DatasetError(
            f"File is {len(content) / 1048576:.1f} MB; the configured limit is {max_upload_mb} MB."
        )
    try:
        if ext == ".csv":
            frame = None
            errors = []
            for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
                try:
                    decoded = content.decode(encoding)
                    original_header = next(csv.reader(io.StringIO(decoded)))
                    frame = pd.read_csv(
                        io.BytesIO(content),
                        encoding=encoding,
                        names=_dedupe_columns(original_header),
                        header=0,
                        on_bad_lines="error",
                        low_memory=False,
                        dtype="string",
                        skipinitialspace=True,
                        nrows=500_001,
                    )
                    break
                except UnicodeDecodeError as exc:
                    errors.append(str(exc))
            if frame is None:
                raise DatasetError(
                    "Could not decode CSV. Save it as UTF-8 and try again."
                )
            if len(frame) > 500_000:
                raise DatasetError("Datasets are limited to 500,000 rows.")
        else:
            book = pd.ExcelFile(io.BytesIO(content))
            if len(book.sheet_names) > 1 and sheet_name is None:
                raise DatasetError(
                    "This workbook has multiple sheets. Select a sheet before loading."
                )
            chosen = sheet_name or book.sheet_names[0]
            if chosen not in book.sheet_names:
                raise DatasetError(
                    f"Sheet '{chosen}' does not exist. Available sheets: {', '.join(book.sheet_names)}"
                )
            frame = pd.read_excel(book, sheet_name=chosen, nrows=500_001)
            if len(frame) > 500_000:
                raise DatasetError("Datasets are limited to 500,000 rows.")
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(
            f"Could not read {ext[1:].upper()} file: {type(exc).__name__}: {exc}"
        ) from exc
    if frame is None or frame.empty or len(frame.columns) == 0:
        raise DatasetError("The file has no data rows or columns.")
    if len(frame.columns) > 500:
        raise DatasetError("Datasets are limited to 500 columns.")
    frame.columns = _dedupe_columns(list(frame.columns))
    frame, types, roles, conversions = _infer_types(frame)
    fingerprint = hashlib.sha256(content).hexdigest()
    info = {
        "name": safe_name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "types": types,
        "roles": roles,
        "conversions": conversions,
        "fingerprint": fingerprint,
        "duplicate_rows": int(frame.duplicated().sum()),
        "sheet": sheet_name,
    }
    return frame, info
