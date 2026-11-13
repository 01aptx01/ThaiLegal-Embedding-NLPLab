"""Dataset loading, normalization, matching, and audit helpers for the RAG run."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATASETS = {
    "corpus": ("VISAI-AI/nitibench-statute", "ccl"),
    "pilot": ("airesearch/WangchanX-Legal-ThaiCCL-RAG", "train"),
    "test": ("VISAI-AI/nitibench", "ccl"),
}

REQUIRED_COLUMNS = {
    "corpus": {"law_name", "section_num", "section_content"},
    "pilot": {"question", "positive_contexts", "positive_answer"},
    "test": {"question", "relevant_laws", "reference_answer"},
}
QUESTION_COLUMNS = ["question_id", "question", "gold_ids", "gold_law", "reference_answer"]
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def search_text(text: Any) -> str:
    """Normalize searchable text using the notebook's matching rules."""
    text = unicodedata.normalize("NFC", str(text) if text is not None else "").translate(THAI_DIGITS)
    return " ".join(text.replace("\u200b", " ").replace("\ufeff", "").split())


def _law_key(name: Any, section: Any) -> tuple[str, str]:
    name_key = "".join(search_text(name).split())
    section_key = "".join(search_text(section).split()).removeprefix("มาตรา")
    return name_key, section_key


def load_sources(api: Any, revisions: dict[str, str] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Load pinned datasets; resolve missing revision pins through the supplied API."""
    from datasets import load_dataset

    requested = revisions or {}
    raw: dict[str, Any] = {}
    resolved: dict[str, str] = {}
    for name, (repo, split) in DATASETS.items():
        requested_ref = requested.get(name)
        if requested_ref and re.fullmatch(r"[0-9a-fA-F]{40}", requested_ref):
            revision = requested_ref
        else:
            info = api.dataset_info(repo, revision=requested_ref) if requested_ref else api.dataset_info(repo)
            revision = info.sha
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise ValueError(f"{name}: dataset revision did not resolve to a 40-character commit SHA")
        resolved[name] = revision
        raw[name] = load_dataset(repo, split=split, revision=revision)
        columns = set(raw[name].column_names)
        missing = REQUIRED_COLUMNS[name] - columns
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        logger.info("Loaded %s: %d rows at revision %s", name, len(raw[name]), revision)
    return raw, resolved


def _gold_entries(row: Mapping[str, Any], split: str):
    if split == "pilot":
        for item in row.get("positive_contexts") or []:
            item = item or {}
            metadata = item.get("metadata") or {}
            yield metadata.get("law_title"), metadata.get("section"), item.get("context")
    else:
        for item in row.get("relevant_laws") or []:
            item = item or {}
            yield item.get("law_name"), item.get("section_num"), item.get("section_content")


def prepare_data(
    raw: Mapping[str, Any], mode: str, pilot_questions: int, seed: int, run_dir: str | Path
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Build ordered statute IDs and eligible questions, writing notebook-compatible audits."""
    if mode not in {"pilot", "test"}:
        raise ValueError('mode must be "pilot" or "test"')
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    corpus: dict[str, dict[str, Any]] = {}
    statute_lookup: dict[tuple[str, str], list[str]] = defaultdict(list)
    corpus_rows = raw["corpus"]
    if len(corpus_rows) == 0:
        raise ValueError("corpus is empty")
    for index, row in enumerate(corpus_rows, 1):
        key = _law_key(row.get("law_name"), row.get("section_num"))
        if not all(key) or not search_text(row.get("section_content")):
            raise ValueError(f"corpus row {index}: missing law, section, or text")
        statute_id = f"S{index:05d}"
        corpus[statute_id] = dict(row)
        statute_lookup[key].append(statute_id)

    duplicate_keys = [
        {"law_key": key[0], "section_key": key[1], "ids": ids}
        for key, ids in statute_lookup.items() if len(ids) > 1
    ]
    pd.DataFrame(duplicate_keys, columns=["law_key", "section_key", "ids"]).to_json(
        run_path / "duplicate_keys.jsonl", orient="records", lines=True, force_ascii=False
    )
    pd.DataFrame([
        {"statute_id": sid, "law_name": row.get("law_name"),
         "section_num": row.get("section_num"), "law_code": row.get("law_code")}
        for sid, row in corpus.items()
    ]).to_csv(run_path / "statute_ids.csv", index=False, encoding="utf-8-sig")

    def match_statute(name: Any, section: Any, text: Any) -> tuple[str | None, str]:
        key = _law_key(name, section)
        candidates = statute_lookup.get(key, []) if all(key) else []
        if len(candidates) == 1:
            return candidates[0], "matched_key"
        if len(candidates) > 1 and search_text(text):
            exact = [sid for sid in candidates
                     if search_text(corpus[sid]["section_content"]) == search_text(text)]
            if len(exact) == 1:
                return exact[0], "matched_text"
        return None, "ambiguous" if candidates else "unmatched"

    test_texts = set()
    for row in raw["test"]:
        normalized_question = search_text(row.get("question"))
        if normalized_question:
            test_texts.add(normalized_question)
    usable: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for split in ("pilot", "test"):
        accepted: list[dict[str, Any]] = []
        for index, row in enumerate(raw[split], 1):
            question_id = f"{split}-{index:05d}"
            question = search_text(row.get("question"))
            entries = list(_gold_entries(row, split))
            ids: list[str] = []
            statuses: list[str] = []
            for name, section, text in entries:
                statute_id, status = match_statute(name, section, text)
                statuses.append(status)
                mapping_rows.append({
                    "split": split, "question_id": question_id, "law_name": name,
                    "section": section, "status": status, "statute_id": statute_id,
                })
                if statute_id:
                    ids.append(statute_id)
            ids = list(dict.fromkeys(ids))
            overlap = split == "pilot" and bool(question) and question in test_texts
            fully_mapped = bool(entries) and all(status.startswith("matched") for status in statuses)
            reason = (
                "blank_question" if not question else
                "test_overlap" if overlap else
                "no_gold" if not entries else
                "gold_not_fully_mapped" if not fully_mapped else "included"
            )
            audit_rows.append({
                "split": split, "question_id": question_id, "reason": reason,
                "gold_entries": len(entries), "mapped_unique": len(ids),
            })
            if reason == "included":
                answer_field = "positive_answer" if split == "pilot" else "reference_answer"
                accepted.append({
                    "question_id": question_id, "question": row["question"], "gold_ids": ids,
                    "gold_law": corpus[ids[0]]["law_name"],
                    "reference_answer": row.get(answer_field) or "",
                })
        usable[split] = pd.DataFrame(accepted, columns=QUESTION_COLUMNS)

    data_audit = pd.DataFrame(audit_rows, columns=[
        "split", "question_id", "reason", "gold_entries", "mapped_unique"
    ])
    mapping_audit = pd.DataFrame(mapping_rows, columns=[
        "split", "question_id", "law_name", "section", "status", "statute_id"
    ])
    data_audit.to_csv(run_path / "question_audit.csv", index=False, encoding="utf-8-sig")
    mapping_audit.to_csv(run_path / "mapping_audit.csv", index=False, encoding="utf-8-sig")
    logger.info("Corpus: %d | normalized duplicate keys: %d", len(corpus), len(duplicate_keys))
    if not data_audit.empty:
        logger.info("Question audit summary:\n%s", pd.crosstab(data_audit.split, data_audit.reason))

    questions = usable[mode].copy()
    if questions.empty:
        raise ValueError("ไม่มีคำถามที่ใช้ได้: ตรวจ mapping_audit.csv")
    if mode == "pilot":
        questions = questions.sample(n=min(pilot_questions, len(questions)), random_state=seed)
    questions = questions.reset_index(drop=True)
    logger.info("Evaluation questions: %d", len(questions))
    return corpus, questions


def select_generation_questions(questions: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Select a reproducible, gold-law-stratified sample for generation evaluation."""
    eligible = questions.loc[questions["reference_answer"].str.strip().ne("")]
    if eligible.empty:
        return eligible.set_index("question_id")
    grouped = eligible.sample(frac=1, random_state=seed).sort_values("gold_law", kind="stable")
    sample_size = min(limit, len(grouped))
    if sample_size <= 0:
        return grouped.iloc[0:0].set_index("question_id")
    positions = ((np.arange(sample_size) + 0.5) * len(grouped) / sample_size).astype(int)
    return grouped.iloc[positions].set_index("question_id")
