"""End-to-end orchestration for pinned, auditable Thai legal RAG runs."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import RunConfig

logger = logging.getLogger(__name__)
PACKAGES = ["datasets", "pythainlp", "gensim", "scikit-learn", "sentence-transformers", "transformers", "torch", "numpy", "pandas", "scipy", "huggingface-hub", "tqdm"]


def _versions() -> dict[str, str | None]:
    result = {}
    for package in PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _resolve_model_revision(api: Any, model_id: str, requested: str | None) -> str:
    """Resolve a model ref to a commit SHA and reject incomplete API responses."""
    sha = requested if requested and re.fullmatch(r"[0-9a-fA-F]{40}", requested) else api.model_info(
        model_id, revision=requested
    ).sha
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise ValueError(f"{model_id}: model revision did not resolve to a 40-character commit SHA")
    return sha


def _archive(run_dir: Path) -> Path:
    """Create/replace the sibling zip atomically, preserving an old zip on failure."""
    final = Path(str(run_dir) + ".zip")
    fd, temporary = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(run_dir.rglob("*")):
                if item.is_file():
                    archive.write(item, item.relative_to(run_dir))
        os.replace(temporary, final)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return final


def run(config: RunConfig) -> Path:
    """Execute the notebook workflow and return the created run directory.

    The directory and a manifest are created before model work so partial failures
    retain their audits and resolved settings. A ZIP is published only on completion.
    """
    import numpy as np
    import torch
    from huggingface_hub import HfApi

    from .data import DATASETS, load_sources, prepare_data, select_generation_questions
    from .retrieval import run_retrieval, write_subset_summary

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
    dtype = torch.float16 if device == "cuda" else torch.float32
    # Preserve the notebook's deterministic seed behavior, now including CUDA.
    import random
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = config.output_dir / datetime.now().strftime(f"{config.mode}_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "mode": config.mode, "seed": config.seed, "datasets": DATASETS,
        "dataset_revisions": {}, "python": platform.python_version(), "versions": _versions(),
        "device": device, "dtype": str(dtype), "retrieval_only": config.retrieval_only,
        "bge": config.bge_id, "bge_revision": None, "bge_max_tokens": config.bge_max_tokens,
        "bge_batch_size": config.bge_batch_size, "generator": config.generator_id,
        "generator_revision": None, "top_k": config.top_k, "rank_k": config.rank_k,
        "max_input_tokens": config.max_input_tokens, "max_new_tokens": config.max_new_tokens,
        "pilot_questions": config.pilot_questions,
        "pilot_generation_questions": config.pilot_generation_questions,
        "test_generation_questions": config.test_generation_questions,
        "requested_revisions": {
            "corpus": config.corpus_revision, "pilot": config.pilot_revision,
            "test": config.test_revision, "bge": config.bge_revision,
            "generator": config.generator_revision,
        },
        "generation_question_ids": [], "stages": {},
    }
    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    revisions = {k: v for k, v in {
        "corpus": config.corpus_revision, "pilot": config.pilot_revision,
        "test": config.test_revision,
    }.items() if v}
    api = HfApi()
    try:
        raw, dataset_revisions = load_sources(api, revisions)
        manifest["dataset_revisions"] = dataset_revisions
        manifest["stages"]["data_loading"] = "complete"
        _write_json(manifest_path, manifest)

        corpus, questions = prepare_data(raw, config.mode, config.pilot_questions, config.seed, run_dir)
        manifest["question_ids"] = questions["question_id"].tolist()
        manifest["stages"]["data_preparation"] = "complete"
        gen_limit = config.pilot_generation_questions if config.mode == "pilot" else config.test_generation_questions
        generation_questions = select_generation_questions(questions, gen_limit, config.seed)
        manifest["generation_question_ids"] = generation_questions.index.tolist()
        manifest["question_count"] = len(questions)
        manifest["generation_question_count"] = len(generation_questions)
        _write_json(manifest_path, manifest)

        bge_revision = _resolve_model_revision(api, config.bge_id, config.bge_revision)
        manifest["bge_revision"] = bge_revision
        retrieval_rows, shared_tokenize_s = run_retrieval(
            corpus, questions, config, bge_revision, device, dtype, run_dir
        )
        manifest["stages"]["retrieval"] = "complete"
        manifest["shared_tokenize_seconds"] = shared_tokenize_s
        _write_json(manifest_path, manifest)
        generation_retrieval = write_subset_summary(retrieval_rows, generation_questions, run_dir)

        # Preserve the notebook's run_config.json fields for downstream review scripts.
        SYSTEM_PROMPT = ""
        if not config.retrieval_only:
            from .generation import SYSTEM_PROMPT
        run_config = {
            "mode": config.mode, "seed": config.seed, "device": device, "dtype": str(dtype),
            "datasets": DATASETS, "dataset_revisions": dataset_revisions,
            "bge": config.bge_id, "bge_revision": bge_revision,
            "bge_max_tokens": config.bge_max_tokens, "bge_batch_size": config.bge_batch_size,
            "generator": config.generator_id, "generator_revision": None,
            "top_k": config.top_k, "rank_k": config.rank_k,
            "max_input_tokens": config.max_input_tokens, "max_new_tokens": config.max_new_tokens,
            "system_prompt": SYSTEM_PROMPT, "generation_question_ids": generation_questions.index.tolist(),
            "versions": manifest["versions"],
        }
        if not config.retrieval_only:
            if generation_questions.empty:
                manifest["stages"]["generation"] = "skipped: no eligible generation questions"
                _write_json(manifest_path, manifest)
                raise ValueError(
                    f"No eligible generation questions were selected. Retrieval and audits are in {run_dir}; "
                    "rerun with retrieval-only or inspect question_audit.csv and mapping_audit.csv."
                )
            from .generation import run_generation
            generator_revision = _resolve_model_revision(api, config.generator_id, config.generator_revision)
            manifest["generator_revision"] = generator_revision
            run_config["generator_revision"] = generator_revision
            _write_json(run_dir / "run_config.json", run_config)
            _write_json(manifest_path, manifest)
            run_generation(
                corpus, generation_questions, generation_retrieval, config,
                generator_revision, device, dtype, run_dir,
            )
            manifest["stages"]["generation"] = "complete"
        else:
            manifest["stages"]["generation"] = "skipped: retrieval-only"
            _write_json(run_dir / "run_config.json", run_config)
        _write_json(manifest_path, manifest)
        archive = _archive(run_dir)
        logger.info("Completed %s run at %s (archive %s)", config.mode, run_dir, archive)
        return run_dir
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        logger.error("Run failed; partial results and audits are at %s: %s", run_dir, exc)
        raise


def review(run_dir: str | Path, scores_path: str | Path | None = None) -> bool:
    """Summarize human scores and refresh the run archive after successful review."""
    from .review import summarize_review

    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {path}")
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stages = manifest.get("stages")
        if stages is not None and stages.get("generation") != "complete":
            raise ValueError(f"Cannot review an incomplete run; generation is not complete: {path}")
    success = summarize_review(path, scores_path)
    _archive(path)
    return success
