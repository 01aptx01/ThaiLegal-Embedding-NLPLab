"""Build and query a reusable, exact-cosine index for the statute corpus.

Index artifacts are intended to be loaded only from a trusted local directory:
the vectorizer and Word2Vec model use joblib/pickle, which can execute code when
loading an untrusted file. The corpus and manifest are plain JSON. BGE uses the
same numeric choice as the evaluation pipeline: float16 on CUDA and float32 on
CPU; the chosen dtype is recorded and checked when loading the index.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import RunConfig
from .data import DATASETS, search_text
from .retrieval import METHODS, _tokenize, _word2vec_average, rank_exact

logger = logging.getLogger(__name__)

def _identity(tokens):
    """Pickleable analyzer matching the notebook's token-list analyzer."""
    return tokens
_NOTEBOOK_LABELS = OrderedDict(zip(("r1", "r2", "r3", "r4"), METHODS, strict=True))


def _device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_corpus(revision: str | None) -> tuple[dict[str, dict[str, Any]], str]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo, split = DATASETS["corpus"]
    revision = HfApi().dataset_info(repo, revision=revision).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("corpus revision did not resolve to a 40-character commit SHA")
    rows = load_dataset(repo, split=split, revision=revision)
    corpus: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows, 1):
        sid = f"S{i:05d}"
        if not search_text(row.get("law_name")) or not search_text(row.get("section_num")) or not search_text(row.get("section_content")):
            raise ValueError(f"corpus row {i}: missing law, section, or text")
        corpus[sid] = {"law_name": row["law_name"], "section_num": row["section_num"], "section_content": row["section_content"]}
    if not corpus:
        raise ValueError("corpus is empty")
    return corpus, revision


def build_index(output_dir: Path, *, corpus_revision: str | None = None,
                bge_revision: str | None = None, device: str = "auto",
                batch_size: int = 8, include_dense: bool = True) -> Path:
    """Build all (or sparse/Word2Vec only) rankers atomically at output_dir."""
    output_dir = Path(output_dir).expanduser().absolute()
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if output_dir.exists():
        raise FileExistsError(f"index destination already exists: {output_dir}")
    device = _device(device)
    config = RunConfig(device=device, bge_revision=bge_revision, bge_batch_size=batch_size)
    corpus, corpus_revision = _load_corpus(corpus_revision)
    bge_revision_resolved = None
    dtype = None
    if include_dense:
        from huggingface_hub import HfApi
        import torch
        from .pipeline import _resolve_model_revision
        bge_revision_resolved = _resolve_model_revision(HfApi(), config.bge_id, bge_revision)
        dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("Building inference index: %d statutes at corpus revision %s", len(corpus), corpus_revision)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        ids = list(corpus)
        texts = [search_text(corpus[sid]["section_content"]) for sid in ids]
        tokens = [_tokenize(text) for text in texts]
        vectorizers = [CountVectorizer(binary=True, analyzer=_identity, dtype=np.float32),
                       TfidfVectorizer(analyzer=_identity, dtype=np.float32)]
        for ix, vectorizer in enumerate(vectorizers, 1):
            logger.info("Fitting %s representation", METHODS[ix - 1])
            matrix = normalize(vectorizer.fit_transform(tokens)).tocsr()
            joblib.dump(vectorizer, temp_dir / f"r{ix}_vectorizer.joblib")
            save_npz(temp_dir / f"r{ix}_matrix.npz", matrix)
        from gensim.models import Word2Vec
        logger.info("Training %s representation", METHODS[2])
        w2v = Word2Vec(sentences=tokens, sg=1, vector_size=150, window=5,
                       min_count=2, epochs=10, workers=1, seed=RunConfig.seed)
        matrix = _word2vec_average(tokens, w2v)
        joblib.dump(w2v, temp_dir / "r3_word2vec.joblib")
        np.save(temp_dir / "r3_matrix.npy", matrix)
        available = ["r1", "r2", "r3"]
        if include_dense:
            from sentence_transformers import SentenceTransformer
            logger.info("Encoding %s with %s at revision %s", len(texts), config.bge_id, bge_revision_resolved)
            encoder = SentenceTransformer(config.bge_id, revision=bge_revision_resolved, device=device,
                                          model_kwargs={"torch_dtype": dtype})
            encoder.max_seq_length = config.bge_max_tokens
            matrix = normalize(encoder.encode(texts, batch_size=batch_size, show_progress_bar=True).astype(np.float32))
            np.save(temp_dir / "r4_matrix.npy", matrix)
            del encoder
            available.append("r4")
        (temp_dir / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
        manifest = {"format_version": 1, "corpus_revision": corpus_revision,
                    "bge_revision": bge_revision_resolved, "bge_model": config.bge_id,
                    "bge_max_tokens": config.bge_max_tokens,
                    "bge_dtype": str(dtype) if dtype is not None else None,
                    "ids": ids, "methods": available, "trusted_local_artifacts":
                    "joblib files use pickle; load only an index directory you trust."}
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # A same-filesystem rename publishes the complete index at once and
        # fails if another builder created the destination in the meantime.
        os.rename(temp_dir, output_dir)
        logger.info("Published inference index to %s", output_dir)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_dir


class InferenceIndex:
    """Loaded corpus and warm representations for arbitrary question search."""
    def __init__(self, directory: Path, device: str = "auto"):
        self.directory = Path(directory)
        self.device = _device(device)
        # Initialize PyThaiNLP/newmm lazily here so its one-time import/setup
        # cost does not land on the first user query.
        _tokenize("ทดสอบระบบ")
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.corpus = json.loads((self.directory / "corpus.json").read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
        if self.manifest.get("format_version") != 1:
            raise ValueError(f"unsupported inference index format version: {self.manifest.get('format_version')!r}")
        methods = self.manifest.get("methods")
        if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or any(m not in _NOTEBOOK_LABELS for m in methods):
            raise ValueError("invalid methods list in inference index manifest")
        self.ids = self.manifest.get("ids")
        if not isinstance(self.ids, list) or not self.ids or len(set(self.ids)) != len(self.ids) or list(self.corpus) != self.ids:
            raise ValueError("index IDs do not match the stored corpus")
        self.corpus_revision = self.manifest["corpus_revision"]
        if not isinstance(self.corpus_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", self.corpus_revision):
            raise ValueError("inference index is missing a pinned corpus revision SHA")
        self.methods = OrderedDict((key, _NOTEBOOK_LABELS[key]) for key in methods)
        self.vectorizers = {key: joblib.load(self.directory / f"{key}_vectorizer.joblib") for key in ("r1", "r2") if key in self.methods}
        self.matrices = {}
        for key in self.methods:
            path = self.directory / f"{key}_matrix.npz"
            self.matrices[key] = load_npz(path) if key in {"r1", "r2"} else np.load(path.with_suffix(".npy"), mmap_mode="r")
            if self.matrices[key].shape[0] != len(self.ids):
                raise ValueError(f"{key} matrix row count does not match index IDs")
        self.w2v = joblib.load(self.directory / "r3_word2vec.joblib") if "r3" in self.methods else None
        self.encoder = None
        if "r4" in self.methods:
            from sentence_transformers import SentenceTransformer
            revision = self.manifest.get("bge_revision")
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError("BGE index is missing a pinned model revision SHA")
            import torch
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.encoder = SentenceTransformer(self.manifest["bge_model"], revision=revision, device=self.device,
                                               model_kwargs={"torch_dtype": dtype})
            self.encoder.max_seq_length = self.manifest["bge_max_tokens"]
            if str(dtype) != self.manifest.get("bge_dtype"):
                raise ValueError("BGE query dtype differs from the index embedding dtype")

    @classmethod
    def load(cls, index_dir: Path, device: str = "auto") -> "InferenceIndex":
        return cls(index_dir, device=device)

    def search(self, question: str, method: str, limit: int = 10) -> list[str]:
        if method not in self.methods:
            raise ValueError(f"unknown or unavailable method {method!r}; choose one of {list(self.methods)}")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []
        key = method
        query = search_text(question)
        if key in {"r1", "r2"}:
            q = normalize(self.vectorizers[key].transform([_tokenize(query)]))
            if q.nnz == 0:
                return []
            query_vectors = q
        elif key == "r3":
            q = _word2vec_average([_tokenize(query)], self.w2v)[0]
            if not np.any(q):
                return []
            query_vectors = q[None, :]
        else:
            q = normalize(self.encoder.encode([query], show_progress_bar=False).astype(np.float32))[0]
            if not np.any(q):
                return []
            query_vectors = q[None, :]
        return rank_exact(query_vectors, self.matrices[key], self.ids, min(limit, len(self.ids)))[0]
