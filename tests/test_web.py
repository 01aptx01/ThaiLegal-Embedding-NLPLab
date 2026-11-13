"""Verify the workspace API and the browser's NDJSON contract without loading models."""

import json
import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None
else:
    from thai_legal_rag.web import create_app


class Index:
    methods = {"r1": "R1 Binary BoW", "r2": "R2 TF-IDF"}
    corpus_revision = "fixture"
    corpus = {"S00001": {"law_name": "กฎหมาย", "section_num": "1", "section_content": "ตัวบท"}}

    def search(self, question, method, limit):
        return ["S00001"]


class Generator:
    model_id = "test-generator"

    def events(self, question, retrieved_ids, corpus, cancelled):
        yield {"type": "delta", "text": "คำตอบ"}
        yield {"type": "result", "answer": "คำตอบ [S00001]", "cited_ids": ["S00001"]}


@unittest.skipUnless(TestClient is not None, "install the web extra to test the HTTP API")
class WorkspaceTests(unittest.TestCase):
    def test_streams_retrieval_before_generation_for_each_selected_method(self):
        with TestClient(create_app(index=Index(), generator=Generator())) as client:
            self.assertEqual(client.get("/api/status").json()["corpus_size"], 1)
            response = client.post("/api/infer", json={
                "question": " คำถาม ", "methods": ["r2", "r1"], "generate": True,
            })
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/x-ndjson", response.headers["content-type"])
        events = [json.loads(line) for line in response.iter_lines()]
        self.assertEqual([event["type"] for event in events], [
            "retrieval", "delta", "result", "retrieval", "delta", "result", "done",
        ])
        self.assertEqual(events[0]["method"], "r2")
        self.assertEqual(events[0]["evidence"][0]["text"], "ตัวบท")
        self.assertEqual(events[5]["cited_ids"], ["S00001"])

    def test_validates_unavailable_methods_and_generation(self):
        with TestClient(create_app(index=Index())) as client:
            self.assertEqual(client.post("/api/infer", json={
                "question": "คำถาม", "methods": ["r3"], "generate": False,
            }).status_code, 422)
            self.assertEqual(client.post("/api/infer", json={
                "question": "คำถาม", "methods": ["r1"], "generate": True,
            }).status_code, 409)
            response = client.post("/api/infer", json={
                "question": "คำถาม", "methods": ["r1"], "generate": False,
            })
        events = [json.loads(line) for line in response.iter_lines()]
        self.assertIsNone(events[1]["answer"])

    def test_evaluation_uses_an_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "retrieval_summary.csv").write_text("method,recall_at_5\nR1,0.8\n")
            (run / "manifest.json").write_text('{"mode":"pilot"}')
            with TestClient(create_app(evaluation_dir=run)) as client:
                self.assertEqual(client.get("/api/evaluate").json(), {
                    "mode": "pilot", "rows": [{"method": "R1", "recall_at_5": "0.8"}],
                })
