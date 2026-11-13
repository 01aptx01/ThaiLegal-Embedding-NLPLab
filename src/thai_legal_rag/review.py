"""Import blind human scores for a completed Thai Legal RAG run.

This module deliberately uses only the standard library so review import can
run after generation without loading the notebook's model dependencies.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCORE_RANGES: dict[str, tuple[int, ...]] = {
    "coverage": (0, 1, 2),
    "contradiction": (0, 1),
    "citation_support": (0, 1),
}
HUMAN_SUMMARY_FIELDS = (
    "method",
    "reviewed_answers",
    "mean_coverage",
    "contradiction_rate",
    "citation_support_rate",
    "answers",
)
STAGE_SUMMARY_FIELDS = ("method", "evidence_stage", "count", "mean_coverage")
logger = logging.getLogger(__name__)


def _read_generation_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            for field in ("review_id", "method", "question_id"):
                if row.get(field) is None or str(row[field]).strip() == "":
                    raise ValueError(f"Missing {field} in {path} at line {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No generation rows found in {path}")
    return rows


def _evidence_stage(row: dict[str, Any]) -> str:
    stage = row.get("evidence_stage")
    if stage not in (None, ""):
        return str(stage)
    top_k = row.get("gold_in_top_k")
    in_prompt = row.get("gold_in_prompt")
    if top_k is None or in_prompt is None:
        return "unknown"
    if not bool(top_k):
        return "retrieval_miss"
    if not bool(in_prompt):
        return "context_omitted"
    return "gold_in_prompt"


def _read_scores(path: Path, known_ids: set[str]) -> dict[str, dict[str, float | None]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"review_id", *SCORE_RANGES}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        scores: dict[str, dict[str, float | None]] = {}
        for line_number, source in enumerate(reader, 2):
            review_id = (source.get("review_id") or "").strip()
            if not review_id:
                raise ValueError(f"Blank review_id in {path} at line {line_number}")
            if review_id in scores:
                raise ValueError(f"Duplicate review_id in {path}: {review_id}")
            if review_id not in known_ids:
                raise ValueError(f"review_id does not belong to this run: {review_id}")
            parsed: dict[str, float | None] = {}
            for column, allowed in SCORE_RANGES.items():
                raw = (source.get(column) or "").strip()
                if raw == "":
                    parsed[column] = None
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"Score {column} must be one of {list(allowed)}; "
                        f"got {raw!r} for {review_id}"
                    ) from exc
                if not math.isfinite(value) or value not in allowed:
                    raise ValueError(
                        f"Score {column} must be one of {list(allowed)}; "
                        f"got {raw!r} for {review_id}"
                    )
                parsed[column] = value
            scores[review_id] = parsed
    return scores


def _mean(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_review(run_dir: Path, scores_path: Path | None = None) -> bool:
    """Import ``review_scored.csv`` and write summaries for complete questions.

    A question is complete when every method has a score row with all three
    rubric values present, matching the notebook's paired analysis. Blank
    cells are accepted in the input file but do not make an answer reviewed.
    Returns ``False`` when no question is complete.
    Missing score files raise ``FileNotFoundError`` so the CLI can report the
    expected location clearly.
    """

    run_dir = Path(run_dir)
    scores_path = Path(scores_path) if scores_path is not None else run_dir / "review_scored.csv"
    generation_path = run_dir / "generation_rows.jsonl"
    generation = _read_generation_rows(generation_path)

    by_id: dict[str, dict[str, Any]] = {}
    methods: list[str] = []
    answer_counts: Counter[str] = Counter()
    for row in generation:
        review_id = str(row["review_id"])
        if review_id in by_id:
            raise ValueError(f"Duplicate review_id in {generation_path}: {review_id}")
        method = str(row["method"])
        by_id[review_id] = row
        answer_counts[method] += 1
        if method not in methods:
            methods.append(method)

    scores = _read_scores(scores_path, set(by_id))
    scored_by_question: dict[Any, dict[str, tuple[dict[str, Any], dict[str, float | None]]]] = defaultdict(dict)
    for review_id, score in scores.items():
        row = by_id[review_id]
        question_id, method = row["question_id"], str(row["method"])
        if method in scored_by_question[question_id]:
            raise ValueError(f"Multiple scored answers for question {question_id!r}, method {method!r}")
        scored_by_question[question_id][method] = (row, score)

    complete_questions = {
        question_id
        for question_id, method_rows in scored_by_question.items()
        if all(
            method in method_rows
            and all(method_rows[method][1][metric] is not None for metric in SCORE_RANGES)
            for method in methods
        )
    }
    matched = [
        scored_by_question[question_id][method]
        for question_id in sorted(complete_questions, key=str)
        for method in methods
    ]

    summary_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    if matched:
        method_records: dict[str, list[tuple[dict[str, Any], dict[str, float | None]]]] = defaultdict(list)
        for row, score in matched:
            method_records[str(row["method"])].append((row, score))
        for method in methods:
            records = method_records[method]
            coverages = [s["coverage"] for _, s in records if s["coverage"] is not None]
            contradictions = [s["contradiction"] for _, s in records if s["contradiction"] is not None]
            citation_support = [s["citation_support"] for _, s in records if s["citation_support"] is not None]
            summary_rows.append({
                "method": method,
                "reviewed_answers": len(records),
                "mean_coverage": _mean(coverages),
                "contradiction_rate": _mean(contradictions),
                "citation_support_rate": _mean(citation_support),
                "answers": answer_counts[method],
            })

        by_stage: dict[tuple[str, str], list[float]] = defaultdict(list)
        stage_counts: Counter[tuple[str, str]] = Counter()
        for row, score in matched:
            method, stage = str(row["method"]), _evidence_stage(row)
            stage_counts[(method, stage)] += 1
            if score["coverage"] is not None:
                by_stage[(method, stage)].append(score["coverage"])
        stage_order: list[tuple[str, str]] = []
        for row, _ in matched:
            key = (str(row["method"]), _evidence_stage(row))
            if key not in stage_order:
                stage_order.append(key)
        for method, stage in stage_order:
            stage_rows.append({
                "method": method,
                "evidence_stage": stage,
                "count": stage_counts[(method, stage)],
                "mean_coverage": _mean(by_stage[(method, stage)]),
            })

    # Always replace prior output, including with a header-only CSV when a new
    # import contains no complete question, so old results cannot look current.
    _write_csv(run_dir / "human_summary.csv", HUMAN_SUMMARY_FIELDS, summary_rows)
    _write_csv(run_dir / "human_evidence_stages.csv", STAGE_SUMMARY_FIELDS, stage_rows)

    complete_method_counts = len(complete_questions)
    assessed_counts = {
        metric: sum(score[metric] is not None for _, score in matched)
        for metric in SCORE_RANGES
    }
    logger.info(
        "Imported %d scored rows; %d complete questions (%d matched method rows) "
        "across %d methods.",
        len(scores), complete_method_counts, len(matched), len(methods),
    )
    logger.info(
        "Assessed rubric counts in matched rows: coverage=%d, contradiction=%d, "
        "citation_support=%d.",
        assessed_counts["coverage"], assessed_counts["contradiction"],
        assessed_counts["citation_support"],
    )
    logger.info(
        "Wrote %d method summaries and %d evidence-stage rows.",
        len(summary_rows), len(stage_rows),
    )
    return bool(complete_questions)
