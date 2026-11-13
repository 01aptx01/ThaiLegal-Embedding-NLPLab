import contextlib
import queue
import sys
import threading
import types
import unittest

from thai_legal_rag import generation
from thai_legal_rag.inference_generation import LiveGenerator


class FakeTensor:
    def __init__(self, length, device=None):
        self.shape = (1, length) if isinstance(length, int) else (1, len(length))
        self.device = device
        self.values = None if isinstance(length, int) else length

    def to(self, device):
        return FakeTensor(self.shape[-1], device)

    def __getitem__(self, index):
        return FakeTensor(self.values or [0])

    def tolist(self):
        return self.values or [0]


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        text = " ".join(message["content"] for message in messages)
        return FakeTensor(len(text.split()))

    def decode(self, tokens, **kwargs):
        return "ตอบตาม [S00001] และ [S99999]"


class InferenceGenerationTests(unittest.TestCase):
    def setUp(self):
        self.previous = {name: sys.modules.get(name) for name in ("torch", "transformers")}
        torch = types.ModuleType("torch")
        torch.is_tensor = lambda value: isinstance(value, FakeTensor)
        torch.ones_like = lambda value: value
        torch.inference_mode = contextlib.nullcontext
        sys.modules["torch"] = torch

        class StoppingCriteria:
            pass

        class StoppingCriteriaList(list):
            pass

        class TextIteratorStreamer:
            def __init__(self, tokenizer, **kwargs):
                self.items = queue.Queue()
                self.timeout = kwargs["timeout"]
                self.tokenizer = tokenizer
                self.skip_prompt = kwargs["skip_prompt"]
                self.skip_special_tokens = kwargs["skip_special_tokens"]
                self.decode_kwargs = {"skip_special_tokens": self.skip_special_tokens}
                self.next_tokens_are_prompt = True
                self.token_cache = []

            def __iter__(self):
                return self

            def __next__(self):
                item = self.items.get(timeout=self.timeout)
                if item is None:
                    raise StopIteration
                return item

            def end(self):
                self.items.put(None)

            def on_finalized_text(self, text, stream_end=False):
                if text:
                    self.items.put(text)
                if stream_end:
                    self.items.put(None)

        transformers = types.ModuleType("transformers")
        transformers.StoppingCriteria = StoppingCriteria
        transformers.StoppingCriteriaList = StoppingCriteriaList
        transformers.TextIteratorStreamer = TextIteratorStreamer
        sys.modules["transformers"] = transformers

    def tearDown(self):
        for name, previous in self.previous.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def test_streams_deltas_and_preserves_invalid_citation_ids(self):
        class Model:
            def generate(self, input_ids, **kwargs):
                self.kwargs = kwargs
                kwargs["streamer"].items.put("ตอบตาม ")
                kwargs["streamer"].items.put("[S00001] และ [S99999]")
                kwargs["streamer"].end()
                return FakeTensor(input_ids.shape[-1] + 4)

        model = Model()
        live = LiveGenerator(model, FakeTokenizer(), max_new_tokens=12)
        corpus = {"S00001": {"section_content": "เนื้อหากฎหมาย"}}
        events = list(live.events("คำถาม", ["S00001"], corpus, threading.Event()))

        self.assertEqual([event["type"] for event in events], ["delta", "delta", "result"])
        result = events[-1]
        self.assertEqual(result["answer"], "ตอบตาม [S00001] และ [S99999]")
        self.assertEqual(result["prompt_ids"], ["S00001"])
        self.assertEqual(result["cited_ids"], ["S00001", "S99999"])
        self.assertEqual(result["invalid_ids"], ["S99999"])
        self.assertEqual(result["output_tokens"], 4)
        self.assertEqual(model.kwargs["max_new_tokens"], 12)
        self.assertTrue(model.kwargs["do_sample"] is False)

    def test_no_evidence_abstains_without_running_model(self):
        class Model:
            def generate(self, *args, **kwargs):
                raise AssertionError("generation should not run")

        live = LiveGenerator(Model(), FakeTokenizer())
        events = list(live.events("คำถาม", [], {}, threading.Event()))
        self.assertEqual(events, [{
            "type": "result",
            "answer": generation.ABSTAIN_TEXT,
            "prompt_ids": [],
            "omitted_ids": [],
            "cited_ids": [],
            "invalid_ids": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "generation_ms": 0.0,
            "ttft_ms": 0.0,
        }])

    def test_pre_cancelled_request_returns_without_events_or_model_call(self):
        class Model:
            def generate(self, *args, **kwargs):
                raise AssertionError("generation should not run")

        cancelled = threading.Event()
        cancelled.set()
        events = list(LiveGenerator(Model(), FakeTokenizer()).events(
            "คำถาม", ["S00001"], {"S00001": {"section_content": "ตัวบท"}}, cancelled,
        ))
        self.assertEqual(events, [])

    def test_thai_without_spaces_streams_and_holds_incomplete_byte_fallback(self):
        class ThaiTokenizer(FakeTokenizer):
            def decode(self, tokens, **kwargs):
                if tokens == [1]:
                    return "\ufffd"
                if tokens == [1, 2]:
                    return "กฎหมาย"
                return "กฎหมาย [S00001]"

        class Model:
            def generate(self, input_ids, **kwargs):
                streamer = kwargs["streamer"]
                streamer.put(FakeTensor([9]))  # prompt tokens are skipped
                streamer.put(FakeTensor([1]))  # incomplete UTF-8 fallback
                streamer.put(FakeTensor([2]))
                streamer.put(FakeTensor([3]))
                streamer.end()
                return FakeTensor(input_ids.shape[-1] + 3)

        events = list(LiveGenerator(Model(), ThaiTokenizer()).events(
            "คำถาม", ["S00001"], {"S00001": {"section_content": "ตัวบท"}},
            threading.Event(),
        ))
        deltas = [event["text"] for event in events if event["type"] == "delta"]
        result = events[-1]
        self.assertEqual("".join(deltas), result["answer"])
        self.assertEqual(result["answer"], "กฎหมาย [S00001]")
        self.assertGreater(len(deltas), 1)

    def test_worker_errors_propagate_to_consumer(self):
        class Model:
            def generate(self, input_ids, **kwargs):
                raise RuntimeError("model failed")

        live = LiveGenerator(Model(), FakeTokenizer())
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            list(live.events(
                "คำถาม", ["S00001"], {"S00001": {"section_content": "มาตรา"}},
                threading.Event(),
            ))


if __name__ == "__main__":
    unittest.main()
