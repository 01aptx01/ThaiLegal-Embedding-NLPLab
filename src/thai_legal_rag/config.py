"""Configuration defaults for the Thai legal RAG command line."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    mode: str = "pilot"
    retrieval_only: bool = False
    output_dir: Path = Path("rag_results")
    device: str = "auto"
    seed: int = 42
    pilot_questions: int = 40
    pilot_generation_questions: int = 4
    test_generation_questions: int = 200
    corpus_revision: str | None = None
    pilot_revision: str | None = None
    test_revision: str | None = None
    bge_revision: str | None = None
    generator_revision: str | None = None
    bge_batch_size: int = 8
    top_k: int = 5
    rank_k: int = 10
    bge_id: str = "BAAI/bge-m3"
    bge_max_tokens: int = 4096
    generator_id: str = "typhoon-ai/llama3.2-typhoon2-1b-instruct"
    max_input_tokens: int = 4096
    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        if self.mode not in {"pilot", "test"}:
            raise ValueError("mode must be 'pilot' or 'test'")
        for name in ("pilot_questions", "pilot_generation_questions", "test_generation_questions", "bge_batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
