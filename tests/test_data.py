import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from thai_legal_rag.data import DATASETS, load_sources, prepare_data, search_text, select_generation_questions


def _raw():
    corpus = [
        {"law_name": "กฎหมาย A", "section_num": "มาตรา ๑", "section_content": "บทบัญญัติหนึ่ง", "law_code": "A"},
        {"law_name": "กฎหมายA", "section_num": "1", "section_content": "บทบัญญัติสอง", "law_code": "A"},
        {"law_name": "กฎหมาย B", "section_num": "2", "section_content": "บทบัญญัติสาม", "law_code": "B"},
    ]
    pilot = [
        {"question": "ถามหนึ่ง", "positive_contexts": [{"metadata": {"law_title": "กฎหมาย A", "section": "๑"}, "context": "บทบัญญัติหนึ่ง"}], "positive_answer": "อ้างอิง"},
        {"question": "ถามซ้ำ", "positive_contexts": [{"metadata": {"law_title": "กฎหมาย A", "section": "1"}, "context": "บทบัญญัติหนึ่ง"}], "positive_answer": "ทับซ้อน"},
        {"question": "ถามสาม", "positive_contexts": [{"metadata": {"law_title": "กฎหมาย B", "section": "2"}, "context": "บทบัญญัติสาม"}], "positive_answer": "ตอบ"},
        {"question": "   ", "positive_contexts": [], "positive_answer": ""},
    ]
    test = [
        {"question": "ถามซ้ำ", "relevant_laws": [{"law_name": "กฎหมาย A", "section_num": "1", "section_content": "บทบัญญัติหนึ่ง"}], "reference_answer": "อ้างอิง"},
        {"question": "ถาม B", "relevant_laws": [{"law_name": "กฎหมาย B", "section_num": "2", "section_content": "บทบัญญัติสาม"}], "reference_answer": ""},
    ]
    return {"corpus": corpus, "pilot": pilot, "test": test}


class DataTests(unittest.TestCase):
    def test_search_text_normalizes_digits_unicode_and_invisible_space(self):
        self.assertEqual(search_text("  มาตรา ๑\u200b๒\ufeff  "), "มาตรา 1 2")

    def test_prepare_data_matches_duplicate_by_exact_text_and_audits_empty_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus, questions = prepare_data(_raw(), "pilot", 40, 42, tmp)
            self.assertEqual(list(corpus), ["S00001", "S00002", "S00003"])
            self.assertEqual(set(questions.question_id), {"pilot-00001", "pilot-00003"})
            self.assertEqual(questions.loc[questions.question_id.eq("pilot-00001"), "gold_ids"].iloc[0], ["S00001"])
            audit = pd.read_csv(Path(tmp) / "question_audit.csv", encoding="utf-8-sig")
            self.assertEqual(audit.set_index("question_id").loc["pilot-00002", "reason"], "test_overlap")
            mapping = pd.read_csv(Path(tmp) / "mapping_audit.csv", encoding="utf-8-sig")
            self.assertEqual(mapping.loc[mapping.question_id.eq("pilot-00001"), "status"].iloc[0], "matched_text")
            duplicates = (Path(tmp) / "duplicate_keys.jsonl").read_text(encoding="utf-8")
            self.assertIn('"ids":["S00001","S00002"]', duplicates)

    def test_empty_mapping_audits_keep_headers(self):
        raw = _raw()
        raw["pilot"] = []
        raw["test"] = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "mapping_audit"):
                prepare_data(raw, "test", 40, 42, tmp)
            self.assertEqual(pd.read_csv(Path(tmp) / "question_audit.csv", encoding="utf-8-sig").columns.tolist(),
                             ["split", "question_id", "reason", "gold_entries", "mapped_unique"])
            self.assertEqual(pd.read_csv(Path(tmp) / "mapping_audit.csv", encoding="utf-8-sig").columns.tolist(),
                             ["split", "question_id", "law_name", "section", "status", "statute_id"])

    def test_generation_sample_is_indexed_and_empty_when_no_answers(self):
        frame = pd.DataFrame(columns=["question_id", "question", "gold_ids", "gold_law", "reference_answer"])
        selected = select_generation_questions(frame, 4, 42)
        self.assertTrue(selected.empty)
        self.assertEqual(selected.index.name, "question_id")

    def test_load_sources_honors_revision_pins_and_checks_schema(self):
        class API:
            def __init__(self):
                self.lookups = []
            def dataset_info(self, repo, **kwargs):
                self.lookups.append((repo, kwargs))
                return type("Info", (), {"sha": "a" * 40})()

        class Dataset:
            def __init__(self, columns):
                self.column_names = columns
            def __len__(self):
                return 1

        required = {name: list(cols) for name, cols in {
            "corpus": {"law_name", "section_num", "section_content"},
            "pilot": {"question", "positive_contexts", "positive_answer"},
            "test": {"question", "relevant_laws", "reference_answer"},
        }.items()}
        calls = []
        def fake_load(repo, split, revision):
            calls.append((repo, split, revision))
            return Dataset(required[next(name for name, config in DATASETS.items() if config == (repo, split))])

        datasets_stub = type("DatasetsStub", (), {"load_dataset": staticmethod(fake_load)})()
        api = API()
        revision_refs = {"corpus": "main", "pilot": "v1.2", "test": "b" * 40}
        with patch.dict(sys.modules, {"datasets": datasets_stub}):
            _, resolved = load_sources(api, revision_refs)
        self.assertEqual(resolved, {name: "a" * 40 for name in DATASETS} | {"test": "b" * 40})
        self.assertEqual(api.lookups, [
            (DATASETS["corpus"][0], {"revision": "main"}),
            (DATASETS["pilot"][0], {"revision": "v1.2"}),
        ])
        self.assertEqual(calls[0][2], "a" * 40)
        self.assertEqual(calls[2][2], "b" * 40)

    def test_load_sources_rejects_unresolved_ref_as_revision(self):
        class API:
            def dataset_info(self, repo, revision=None):
                return type("Info", (), {"sha": revision})()

        datasets_stub = type("DatasetsStub", (), {"load_dataset": staticmethod(lambda **kwargs: None)})()
        with patch.dict(sys.modules, {"datasets": datasets_stub}):
            with self.assertRaisesRegex(ValueError, "40-character commit SHA"):
                load_sources(API(), {"corpus": "moving-branch"})

    def test_empty_corpus_fails_clearly(self):
        raw = _raw()
        raw["corpus"] = []
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "corpus is empty"):
                prepare_data(raw, "test", 40, 42, tmp)


if __name__ == "__main__":
    unittest.main()
