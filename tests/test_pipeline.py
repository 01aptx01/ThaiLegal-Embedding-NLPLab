"""Verify the file-based handoff from generation to blind review."""

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from thai_legal_rag.cli import main


class ReviewCommandTests(unittest.TestCase):
    def test_review_refreshes_archive_without_reloading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot.run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"stages": {"generation": "complete"}}), encoding="utf-8"
            )
            records = [
                {
                    "review_id": f"A{i}", "method": f"R{i}", "question_id": "pilot-00001",
                    "gold_in_top_k": 1, "gold_in_prompt": 1,
                }
                for i in range(1, 5)
            ]
            (run_dir / "generation_rows.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            scores_path = run_dir / "review_scored.csv"
            with scores_path.open("w", encoding="utf-8", newline="") as scores:
                writer = csv.writer(scores)
                writer.writerow(["review_id", "coverage", "contradiction", "citation_support"])
                for record in records:
                    writer.writerow([record["review_id"], 2, 0, 1])

            self.assertEqual(main(["review", str(run_dir)]), 0)
            with zipfile.ZipFile(run_dir.with_name(run_dir.name + ".zip")) as archive:
                self.assertIn("human_evidence_stages.csv", archive.namelist())
                lines = archive.read("human_summary.csv").decode("utf-8-sig").splitlines()
                self.assertEqual(len(list(csv.DictReader(lines))), 4)

            scores_path.write_text(
                "review_id,coverage,contradiction,citation_support\nA1,,,\n", encoding="utf-8"
            )
            self.assertEqual(main(["review", str(run_dir)]), 0)
            with zipfile.ZipFile(run_dir.with_name(run_dir.name + ".zip")) as archive:
                lines = archive.read("human_summary.csv").decode("utf-8-sig").splitlines()
                self.assertEqual(list(csv.DictReader(lines)), [])

    def test_review_refuses_partial_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pilot"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"stages": {"retrieval": "complete"}}), encoding="utf-8"
            )
            self.assertEqual(main(["review", str(run_dir)]), 1)
            self.assertFalse((run_dir / "human_summary.csv").exists())
            self.assertFalse(run_dir.with_suffix(".zip").exists())
