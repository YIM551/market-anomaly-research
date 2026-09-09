"""Offline preflight added in 2026; never skips or changes historical news rows."""
import argparse
import csv
from datetime import date
import io
import json
from pathlib import Path
import re

# Exact request windows derived from legacy_experiment.py (not a market calendar).
WINDOWS = {
    "AEMD": ("2018-01-22", "2021-02-07"),
    "APPB": ("2018-03-25", "2021-04-13"),
    "GME": ("2019-01-11", "2022-01-29"),
    "NBDR": ("2018-03-11", "2021-04-03"),
    "TRBO": ("2018-03-30", "2021-04-09"),
}
REQUIRED = ("Date", "Headline", "Summary")


def iso_date(value):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Expected YYYY-MM-DD")
    return date.fromisoformat(value)


def inspect_csv(path, start, end, encoding="utf-8-sig"):
    """Return aggregate diagnostics without printing headlines, paths or raw values."""
    start, end = iso_date(start), iso_date(end)
    if start >= end:
        raise ValueError("Window must have start < end; end is exclusive")
    result = {"file": Path(path).name, "encoding": encoding,
              "window_start": start.isoformat(), "window_end_exclusive": end.isoformat(),
              "records": 0, "valid_width_records": 0, "date_min": None, "date_max": None,
              "unique_dates": 0, "records_in_window": 0, "errors": [], "warnings": []}
    errors = result["errors"]
    try:
        text = Path(path).read_bytes().decode(encoding)
    except (OSError, UnicodeError, LookupError) as exc:
        errors.append({"code": "unreadable_input", "reason": type(exc).__name__})
        result["ok"] = False
        return result
    rows = csv.reader(io.StringIO(text), strict=True)
    try:
        header = next(rows)
    except StopIteration:
        errors.append({"code": "empty_file"})
        result["ok"] = False
        return result
    except csv.Error:
        errors.append({"code": "invalid_csv"})
        result["ok"] = False
        return result
    if len(set(header)) != len(header):
        errors.append({"code": "duplicate_header"})
    missing = [name for name in REQUIRED if name not in header]
    if missing:
        errors.append({"code": "missing_columns", "required": missing})
    # Case-insensitive Date is inspected only to report coverage. Schema still fails.
    date_index = header.index("Date") if "Date" in header else (
        header.index("date") if "date" in header else None)
    dates, seen = [], set()
    try:
        for record, row in enumerate(rows, start=2):
            result["records"] += 1
            if len(row) != len(header):
                errors.append({"code": "column_count", "record": record,
                               "expected": len(header), "actual": len(row)})
                continue
            result["valid_width_records"] += 1
            if tuple(row) in seen:
                result["warnings"].append({"code": "duplicate_record", "record": record})
            seen.add(tuple(row))
            if date_index is not None:
                try:
                    day = iso_date(row[date_index])
                    dates.append(day)
                    if start <= day < end:
                        result["records_in_window"] += 1
                except ValueError:
                    errors.append({"code": "invalid_date", "record": record})
            if not missing and not (row[header.index("Headline")].strip() or row[header.index("Summary")].strip()):
                errors.append({"code": "empty_text", "record": record})
    except csv.Error:
        errors.append({"code": "invalid_csv", "physical_line": rows.line_num})
    if not result["records"]:
        errors.append({"code": "no_records"})
    if dates:
        result.update(date_min=min(dates).isoformat(), date_max=max(dates).isoformat(), unique_dates=len(set(dates)))
        if not result["records_in_window"]:
            errors.append({"code": "no_date_overlap"})
        elif result["records_in_window"] < len(dates):
            result["warnings"].append({"code": "dates_outside_window",
                                        "count": len(dates)-result["records_in_window"]})
    result["ok"] = not errors
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ticker", choices=list(WINDOWS), nargs="+", default=list(WINDOWS))
    parser.add_argument("--encoding", default="utf-8-sig", help="Explicit input encoding; no guessing")
    args = parser.parse_args(argv)
    reports = [dict(ticker=ticker, **inspect_csv(args.data_dir / f"{ticker}_augmented.csv", *WINDOWS[ticker], args.encoding))
               for ticker in args.ticker]
    output = {"check": "offline_news_preflight", "historical_training_reproduced": False,
              "ok": all(r["ok"] for r in reports), "files": reports}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
