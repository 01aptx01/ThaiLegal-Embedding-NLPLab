"""Warm, token-streaming inference for the Thai legal RAG demo.

Transformers and torch are imported only when a model is loaded, so the
inference API remains importable in lightweight environments.
"""

from __future__ import annotations

import queue
import threading
from time import perf_counter
from typing import Any, Mapping, Sequence

from .generation import ABSTAIN_TEXT, _CITATION_RE, _make_prompt


DEFAULT_MODEL_ID = "typhoon-ai/llama3.2-typhoon2-1b-instruct"
DEFAULT_MAX_INPUT_TOKENS = 4096


class LiveGenerator:
    """A loaded causal language model that streams decoded answer chunks."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str = DEFAULT_MODEL_ID,
        device: Any = "cpu",
        max_new_tokens: int = 256,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
        revision: str | None = None,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be at least 1")
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.revision = revision
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

    @classmethod
    def load(
        cls,
        device: Any = "auto",
        revision: str | None = None,
        max_new_tokens: int = 256,
    ) -> "LiveGenerator":
        """Load the default Typhoon instruct checkpoint onto the selected device."""
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .pipeline import _resolve_model_revision

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = DEFAULT_MODEL_ID
        resolved_revision = _resolve_model_revision(HfApi(), model_id, revision)
        kwargs = {"revision": resolved_revision}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            **kwargs,
        ).to(device).eval()
        return cls(model, tokenizer, model_id, device, max_new_tokens, revision=resolved_revision)

    def events(
        self,
        question: str,
        retrieved_ids: Sequence[str],
        corpus: Mapping[str, Mapping[str, Any]],
        cancelled: threading.Event,
    ):
        """Yield text deltas and a final result event for one question."""
        if cancelled.is_set():
            return
        input_ids, included, omitted = _make_prompt(
            question,
            retrieved_ids,
            corpus,
            self.tokenizer,
            self.max_input_tokens,
            self.device,
        )
        input_tokens = int(input_ids.shape[-1])
        if not included:
            yield {
                "type": "result",
                "answer": ABSTAIN_TEXT,
                "prompt_ids": included,
                "omitted_ids": omitted,
                "cited_ids": [],
                "invalid_ids": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "generation_ms": 0.0,
                "ttft_ms": 0.0,
            }
            return

        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        class CancelOnEvent(StoppingCriteria):
            def __call__(self, _input_ids: Any, _scores: Any, **_kwargs: Any) -> bool:
                return cancelled.is_set()

        class ThaiTextIteratorStreamer(TextIteratorStreamer):
            """Decode the whole token prefix but publish stable new text eagerly.

            Unlike TextStreamer’s space-based word buffering, this works for
            Thai text without spaces. Decoding the accumulated token IDs keeps
            SentencePiece byte fallback intact; replacement characters are
            held until the byte sequence is complete.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._decoded = ""
                self._emitted = ""

            def put(self, value: Any) -> None:
                if len(value.shape) > 1:
                    if value.shape[0] > 1:
                        raise ValueError("TextStreamer only supports batch size 1")
                    value = value[0]
                if self.skip_prompt and self.next_tokens_are_prompt:
                    self.next_tokens_are_prompt = False
                    return
                self.token_cache.extend(value.tolist())
                decoded = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
                self._decoded = decoded
                if decoded.startswith(self._emitted):
                    stable = decoded[:-1] if decoded.endswith("\ufffd") else decoded
                    piece = stable[len(self._emitted):]
                    if piece:
                        self._emitted = stable
                        self.on_finalized_text(piece)

            def end(self) -> None:
                # Flush any held final character at stream completion.
                piece = self._decoded[len(self._emitted):] if self._decoded.startswith(self._emitted) else ""
                self.token_cache = []
                self.next_tokens_are_prompt = True
                self.on_finalized_text(piece, stream_end=True)

        streamer = ThaiTextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.1,
        )
        worker_errors: list[Exception] = []
        output_tokens = 0
        generated_answer: str | None = None
        answer_parts: list[str] = []
        first_token_at: float | None = None
        generation_started = 0.0

        def generate() -> None:
            nonlocal generation_started, output_tokens, generated_answer
            try:
                generation_started = perf_counter()
                with torch.inference_mode():
                    output = self.model.generate(
                        input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        do_sample=False,
                        use_cache=True,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList([CancelOnEvent()]),
                    )
                if output is not None:
                    output_tokens = max(0, int(output.shape[-1]) - input_tokens)
                    generated_answer = self.tokenizer.decode(
                        output[0, input_tokens:], **streamer.decode_kwargs,
                    ).strip()
            except Exception as exc:
                worker_errors.append(exc)
                # Ensure the consumer wakes even when generate fails before the
                # streamer receives its normal end-of-stream marker.
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=generate, name="thai-legal-generation")
        thread_started = False
        try:
            thread.start()
            thread_started = True
            while True:
                try:
                    piece = next(streamer)
                except StopIteration:
                    break
                except (queue.Empty, TimeoutError):
                    if not thread.is_alive():
                        break
                    continue
                if piece:
                    if first_token_at is None:
                        first_token_at = perf_counter()
                    answer_parts.append(piece)
                    yield {"type": "delta", "text": piece}
            thread.join()
            if worker_errors:
                raise worker_errors[0]
        finally:
            # Closing a consumer early also stops generation at the next model
            # step and ensures the background thread is not left behind.
            if thread_started and thread.is_alive():
                cancelled.set()
                thread.join()

        # Stream deltas are best-effort incremental text. Decode the returned
        # token suffix once for the authoritative answer and citation audit.
        answer = generated_answer if generated_answer is not None else "".join(answer_parts).strip()
        cited_ids = list(dict.fromkeys(_CITATION_RE.findall(answer)))
        invalid_ids = [sid for sid in cited_ids if sid not in included]
        finished = perf_counter()
        yield {
            "type": "result",
            "answer": answer,
            "prompt_ids": included,
            "omitted_ids": omitted,
            "cited_ids": cited_ids,
            "invalid_ids": invalid_ids,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "generation_ms": max(0.0, (finished - generation_started) * 1000),
            "ttft_ms": (
                max(0.0, (first_token_at - generation_started) * 1000)
                if first_token_at is not None else None
            ),
        }
