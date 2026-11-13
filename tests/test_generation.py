import contextlib
import csv
import json
import sys
import tempfile
import types
import unittest

import pandas as pd
from thai_legal_rag import generation


class FakeTensor:
    def __init__(self, length, device=None):
        self.shape = (1, length)
        self.device = device

    def to(self, device):
        return FakeTensor(self.shape[-1], device)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return [11, 12]
        return self

    def __len__(self):
        return self.shape[-1]


def install_fake_torch(monkeypatch):
    fake = types.ModuleType("torch")
    fake.is_tensor = lambda value: isinstance(value, FakeTensor)
    fake.ones_like = lambda value: value
    fake.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        text = " ".join(message["content"] for message in messages)
        return FakeTensor(len(text.split()))

    def decode(self, tokens, skip_special_tokens=True):
        return "คำตอบตาม [S00002] และอ้างเกิน [S99999]"


class GenerationTests(unittest.TestCase):
  def setUp(self):
    self.previous_torch = sys.modules.get("torch")
    self.previous_transformers = sys.modules.get("transformers")
    self.previous_tqdm = sys.modules.get("tqdm")
    self.previous_tqdm_auto = sys.modules.get("tqdm.auto")
    install_fake_torch(self)
    tqdm = types.ModuleType("tqdm")
    tqdm_auto = types.ModuleType("tqdm.auto")
    tqdm_auto.tqdm = lambda iterable, **kwargs: iterable
    self.setitem(sys.modules, "tqdm", tqdm)
    self.setitem(sys.modules, "tqdm.auto", tqdm_auto)

  def tearDown(self):
    for name, previous in (
      ("torch", self.previous_torch),
      ("transformers", self.previous_transformers),
      ("tqdm", self.previous_tqdm),
      ("tqdm.auto", self.previous_tqdm_auto),
    ):
      if previous is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = previous

  def setitem(self, mapping, key, value):
    mapping[key] = value

  def test_prompt_skips_oversized_statute_and_keeps_later_one(self):
    corpus = {
        "S00001": {"section_content": "ยาวมาก " * 30},
        "S00002": {"section_content": "กฎข้อที่สอง"},
    }
    tokenizer = FakeTokenizer()
    question = "คำถาม"
    base_length = generation._prompt_ids(question, [], tokenizer).shape[-1]
    second_length = generation._prompt_ids(
        question, ["[S00002] กฎข้อที่สอง"], tokenizer,
    ).shape[-1]

    ids, included, omitted = generation._make_prompt(
        question, ["S00001", "S00002"], corpus, tokenizer, second_length, "cpu",
    )

    self.assertEqual(ids.shape[-1], second_length)
    self.assertEqual(included, ["S00002"])
    self.assertEqual(omitted, ["S00001"])
    self.assertGreater(second_length, base_length)

  def test_prompt_rejects_non_tensor_chat_template(self):
    class BadTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]

    with self.assertRaisesRegex(TypeError, "torch tensor"):
        generation._prompt_ids("คำถาม", [], BadTokenizer())

  def test_run_generation_writes_jsonl_summary_and_review_artifacts(self):

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, input_ids, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 5
            return FakeTensor(input_ids.shape[-1] + 2)

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer(),
    )
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeModel(),
    )
    self.setitem(sys.modules, "transformers", transformers)

    corpus = {
        "S00002": {"section_content": "ตัวบทที่สอง", "law_name": "กฎหมาย", "section_num": "2"},
        "S99999": {"section_content": "ยาว " * 20, "law_name": "อื่น", "section_num": "9"},
    }
    questions = pd.DataFrame(
        {"reference_answer": ["คำตอบอ้างอิง"]}, index=pd.Index(["Q1"], name="question_id"),
    )
    retrieval = pd.DataFrame([{
        "method": "bm25", "question_id": "Q1", "question": "ถามอะไร",
        "retrieved_ids": ["S00002", "S99999"], "gold_ids": ["S00002"],
    }])
    config = types.SimpleNamespace(
        generator_id="fake/model", seed=7, top_k=2, max_input_tokens=100,
        max_new_tokens=5,
    )
    config.max_input_tokens = generation._prompt_ids(
        "ถามอะไร", ["[S00002] ตัวบทที่สอง"], FakeTokenizer(),
    ).shape[-1]
    with tempfile.TemporaryDirectory() as temp_dir:
      run_dir = __import__("pathlib").Path(temp_dir) / "run-demo"
      generation.run_generation(corpus, questions, retrieval, config, "rev1", "cpu", "float32", run_dir)

      rows = [json.loads(line) for line in (run_dir / "generation_rows.jsonl").read_text(encoding="utf-8").splitlines()]
      row = rows[0]
      self.assertEqual(row["prompt_ids"], ["S00002"])
      self.assertEqual(row["cited_ids"], ["S00002", "S99999"])
      self.assertEqual(row["invalid_ids"], ["S99999"])
      self.assertEqual(row["gold_cited"], 1)
      self.assertEqual(row["reference_answer"], "คำตอบอ้างอิง")

      with (run_dir / "generation_summary.csv").open(encoding="utf-8") as f:
        summary = next(csv.DictReader(f))
      self.assertEqual(summary["method"], "bm25")
      self.assertEqual(summary["answers"], "1")
      with (run_dir / "evidence_stages.csv").open(encoding="utf-8") as f:
        stages = next(csv.DictReader(f))
      self.assertEqual(stages["method"], "bm25")
      self.assertEqual(stages["gold_in_prompt"], "1")

      review_bytes = (run_dir / "review_template.csv").read_bytes()
      self.assertTrue(review_bytes.startswith(b"\xef\xbb\xbf"))
      review = pd.read_csv(run_dir / "review_template.csv", encoding="utf-8-sig")
      self.assertEqual(list(review.columns), [
        "review_id", "question_id", "question", "reference_answer", "answer", "cited_ids",
        "context_text", "coverage", "contradiction", "citation_support", "note",
      ])
