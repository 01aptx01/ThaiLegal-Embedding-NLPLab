import csv
import json
import tempfile
import unittest
from pathlib import Path

from thai_legal_rag.review import summarize_review


class SummarizeReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)
        self.rows = [
            {"review_id": "A1", "method": "exact", "question_id": "Q1",
             "gold_in_top_k": 1, "gold_in_prompt": 1},
            {"review_id": "A2", "method": "semantic", "question_id": "Q1",
             "gold_in_top_k": 1, "gold_in_prompt": 0},
            {"review_id": "A3", "method": "exact", "question_id": "Q2",
             "gold_in_top_k": 0, "gold_in_prompt": 0},
            {"review_id": "A4", "method": "semantic", "question_id": "Q2",
             "gold_in_top_k": 1, "gold_in_prompt": 1},
        ]
        self.write_generation(self.rows)

    def write_generation(self, rows):
        with (self.run_dir / "generation_rows.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def write_scores(self, rows, path=None):
        path = path or self.run_dir / "review_scored.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "review_id", "coverage", "contradiction", "citation_support", "note",
            ])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read_csv(self, name):
        with (self.run_dir / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_requires_all_three_scores_for_every_method_in_a_question(self):
        self.write_scores([
            {"review_id": "A1", "coverage": "2", "contradiction": "0", "citation_support": "1"},
            # Missing citation_support leaves Q1 incomplete, even though this
            # row's other scores are present.
            {"review_id": "A2", "coverage": "1", "contradiction": "1", "citation_support": ""},
            {"review_id": "A3", "coverage": "0", "contradiction": "0", "citation_support": "0"},
            {"review_id": "A4", "coverage": "2", "contradiction": "1", "citation_support": "1"},
        ])

        with self.assertLogs("thai_legal_rag.review", level="INFO") as logs:
            self.assertTrue(summarize_review(self.run_dir))
        summary = self.read_csv("human_summary.csv")
        self.assertEqual([r["method"] for r in summary], ["exact", "semantic"])
        self.assertEqual(summary[0]["reviewed_answers"], "1")
        self.assertEqual(summary[0]["mean_coverage"], "0.0")
        self.assertEqual(summary[0]["answers"], "2")
        self.assertEqual(summary[1]["reviewed_answers"], "1")
        self.assertEqual(summary[1]["mean_coverage"], "2.0")
        self.assertEqual(summary[1]["contradiction_rate"], "1.0")
        self.assertIn("citation_support=2", " ".join(logs.output))

        stages = self.read_csv("human_evidence_stages.csv")
        self.assertEqual(
            {(r["method"], r["evidence_stage"], r["count"]) for r in stages},
            {("exact", "retrieval_miss", "1"), ("semantic", "gold_in_prompt", "1")},
        )

    def test_all_blank_template_does_not_count_as_reviewed(self):
        self.write_scores([
            {"review_id": row["review_id"], "coverage": "", "contradiction": "", "citation_support": ""}
            for row in self.rows
        ])
        self.assertFalse(summarize_review(self.run_dir))
        self.assertEqual(self.read_csv("human_summary.csv"), [])

    def test_no_complete_questions_overwrites_stale_summary_with_header_only(self):
        (self.run_dir / "human_summary.csv").write_text("old,stale\n", encoding="utf-8")
        self.write_scores([{"review_id": "A1", "coverage": "2", "contradiction": "0", "citation_support": "1"}])

        self.assertFalse(summarize_review(self.run_dir))
        with (self.run_dir / "human_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            self.assertEqual(next(reader), ["method", "reviewed_answers", "mean_coverage",
                                             "contradiction_rate", "citation_support_rate", "answers"])
            self.assertEqual(list(reader), [])

    def test_rejects_duplicate_and_foreign_review_ids(self):
        self.write_scores([
            {"review_id": "A1", "coverage": "", "contradiction": "", "citation_support": ""},
            {"review_id": "A1", "coverage": "", "contradiction": "", "citation_support": ""},
        ])
        with self.assertRaisesRegex(ValueError, "Duplicate review_id"):
            summarize_review(self.run_dir)

        self.write_scores([{"review_id": "outsider", "coverage": "", "contradiction": "", "citation_support": ""}])
        with self.assertRaisesRegex(ValueError, "does not belong"):
            summarize_review(self.run_dir)

    def test_rejects_out_of_range_scores(self):
        self.write_scores([{"review_id": "A1", "coverage": "3", "contradiction": "0", "citation_support": "1"}])
        with self.assertRaisesRegex(ValueError, "coverage must be one of"):
            summarize_review(self.run_dir)

    def test_missing_score_file_is_actionable(self):
        with self.assertRaises(FileNotFoundError):
            summarize_review(self.run_dir)


if __name__ == "__main__":
    unittest.main()
