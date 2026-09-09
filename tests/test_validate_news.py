"""Synthetic contract tests. No external prices, model downloads or training."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_news.py"
spec = importlib.util.spec_from_file_location("validate_news", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NewsValidationTests(unittest.TestCase):
    def inspect(self, content):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "GME_augmented.csv"
            path.write_text(content, encoding="utf-8")
            return module.inspect_csv(path, "2020-01-01", "2022-01-01")

    @staticmethod
    def codes(result):
        return {error["code"] for error in result["errors"]}

    def test_multiline_quoted_csv_is_one_record(self):
        r = self.inspect('Date,Headline,Summary\n2021-01-01,"Example, headline","Line one\nLine two"\n')
        self.assertTrue(r["ok"])
        self.assertEqual((r["records"], r["unique_dates"]), (1, 1))

    def test_lowercase_schema_fails_but_date_coverage_is_reported(self):
        r = self.inspect("date,headline,summary\n2024-01-01,Synthetic,Example\n")
        self.assertEqual(self.codes(r), {"missing_columns", "no_date_overlap"})
        self.assertEqual(r["date_min"], "2024-01-01")

    def test_extra_column_is_not_silently_accepted(self):
        r = self.inspect("Date,Headline,Summary\n2021-01-01,Example,Text,EXTRA\n")
        self.assertIn("column_count", self.codes(r))
        self.assertEqual(r["valid_width_records"], 0)

    def test_short_row_fails(self):
        self.assertIn("column_count", self.codes(self.inspect("Date,Headline,Summary\n2021-01-01,Example\n")))

    def test_duplicate_header_fails(self):
        self.assertIn("duplicate_header", self.codes(self.inspect("Date,Headline,Summary,Date\n2021-01-01,A,B,2021-01-01\n")))

    def test_impossible_and_non_iso_dates_fail(self):
        for day in ["2021-02-29", "01/02/2021", "20210101"]:
            self.assertIn("invalid_date", self.codes(self.inspect(f"Date,Headline,Summary\n{day},A,B\n")))

    def test_empty_file_and_header_only_fail(self):
        self.assertIn("empty_file", self.codes(self.inspect("")))
        self.assertIn("no_records", self.codes(self.inspect("Date,Headline,Summary\n")))

    def test_missing_text_fails(self):
        self.assertIn("empty_text", self.codes(self.inspect("Date,Headline,Summary\n2021-01-01,,\n")))

    def test_end_date_exclusive(self):
        r = self.inspect("Date,Headline,Summary\n2022-01-01,A,B\n")
        self.assertIn("no_date_overlap", self.codes(r))

    def test_partial_overlap_is_warning(self):
        r = self.inspect("Date,Headline,Summary\n2021-01-01,A,B\n2024-01-01,C,D\n")
        self.assertTrue(r["ok"])
        self.assertEqual(r["records_in_window"], 1)
        self.assertEqual(r["warnings"], [{"code": "dates_outside_window", "count": 1}])

    def test_diagnostics_do_not_expose_text(self):
        r = self.inspect("Date,Headline,Summary\ninvalid,PRIVATE_SENTINEL,DO_NOT_ECHO\n")
        serialized = json.dumps(r)
        self.assertNotIn("PRIVATE_SENTINEL", serialized)
        self.assertNotIn("DO_NOT_ECHO", serialized)

    def test_unclosed_quote_fails(self):
        self.assertIn("invalid_csv", self.codes(self.inspect('Date,Headline,Summary\n2021-01-01,"A,B\n')))

    def test_missing_file_and_bad_encoding_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "GME_augmented.csv"
            self.assertIn("unreadable_input", self.codes(module.inspect_csv(path, "2020-01-01", "2022-01-01")))
            path.write_bytes(b"\xff\xfe\xff")
            self.assertIn("unreadable_input", self.codes(module.inspect_csv(path, "2020-01-01", "2022-01-01")))

    def test_cli_returns_failure_and_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            completed = subprocess.run([sys.executable, str(SCRIPT), "--data-dir", folder, "--ticker", "GME"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 1)
            self.assertFalse(json.loads(completed.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
