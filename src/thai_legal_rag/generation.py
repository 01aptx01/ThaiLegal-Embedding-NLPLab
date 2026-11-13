"""Answer generation and review artifacts for Thai legal RAG runs.

Heavy model dependencies are imported only by :func:`run_generation`, so the
module can be imported in lightweight/offline environments.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence


ABSTAIN_TEXT = "หลักฐานไม่เพียงพอ"
SYSTEM_PROMPT = (
    "ตอบคำถามกฎหมายเป็นภาษาไทยโดยใช้เฉพาะตัวบทที่ให้มา "
    "สรุปคำตอบและระบุเงื่อนไขสำคัญให้ชัดเจน "
    "คัดลอกรหัสมาตราพร้อมวงเล็บเหลี่ยมจากตัวบท วางทีละรหัสหลังข้อความที่มาตรานั้นรองรับ "
    "ใช้เฉพาะรหัสที่มีในตัวบท ถ้าหลักฐานไม่พอให้ขึ้นต้นด้วย 'หลักฐานไม่เพียงพอ'"
)

_CITATION_RE = re.compile(r"\[(S\d{5})\]", flags=re.ASCII)


def _prompt_ids(question: str, blocks: Sequence[str], tokenizer: Any) -> Any:
    evidence = "\n\n".join(blocks) or "(ไม่มีตัวบท)"
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"คำถาม\n{question}\n\nตัวบท\n{evidence}"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # Hugging Face's chat template can return a token list when return_tensors
    # is omitted or ignored. Fail early instead of later on .shape/.to calls.
    import torch

    if not torch.is_tensor(input_ids):
        raise TypeError("chat template must return input IDs as a torch tensor")
    return input_ids


def _make_prompt(
    question: str,
    retrieved_ids: Sequence[str],
    corpus: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    max_input_tokens: int,
    device: Any,
) -> tuple[Any, list[str], list[str]]:
    """Build a bounded prompt, omitting whole statutes that do not fit.

    Oversized statutes are skipped while later, shorter statutes are still
    considered. Statute text is never truncated.
    """
    blocks: list[str] = []
    included: list[str] = []
    omitted: list[str] = []
    input_ids = _prompt_ids(question, blocks, tokenizer)
    if input_ids.shape[-1] > max_input_tokens:
        raise ValueError("คำถามและคำสั่งยาวเกินงบ prompt")
    for sid in retrieved_ids:
        block = f"[{sid}] {corpus[sid]['section_content']}"
        candidate = _prompt_ids(question, [*blocks, block], tokenizer)
        if candidate.shape[-1] <= max_input_tokens:
            blocks.append(block)
            included.append(sid)
            input_ids = candidate
        else:
            omitted.append(sid)
    return input_ids.to(device), included, omitted


def _answer(model: Any, tokenizer: Any, input_ids: Any, max_new_tokens: int) -> tuple[str, int, float]:
    import torch

    started = perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, input_ids.shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, len(new_tokens), perf_counter() - started


def _summary_artifacts(records: list[dict[str, Any]], run_dir: Path, seed: int,
                       corpus: Mapping[str, Mapping[str, Any]]) -> None:
    import numpy as np
    import pandas as pd

    generation = pd.DataFrame(records)
    generation["has_citation"] = generation["total_citations"].gt(0)
    generation["evidence_stage"] = np.select(
        [generation["gold_in_top_k"].eq(0), generation["gold_in_prompt"].eq(0)],
        ["retrieval_miss", "context_omitted"],
        default="gold_in_prompt",
    )
    summary = generation.groupby("method", sort=False).agg(
        answers=("review_id", "size"),
        citation_rate=("has_citation", "mean"),
        valid_citations=("valid_citations", "sum"),
        total_citations=("total_citations", "sum"),
        gold_citation_rate=("gold_cited", "mean"),
        gold_in_prompt_rate=("gold_in_prompt", "mean"),
        mean_context_recall=("context_recall", "mean"),
        abstention_rate=("abstained", "mean"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
        output_token_limit_rate=("output_token_limit_reached", "mean"),
        mean_generation_seconds=("generation_seconds", "mean"),
    )
    summary["citation_validity"] = (
        summary["valid_citations"] / summary["total_citations"].replace(0, np.nan)
    )
    summary.to_csv(run_dir / "generation_summary.csv")

    stages = pd.crosstab(generation["method"], generation["evidence_stage"])
    stages.to_csv(run_dir / "evidence_stages.csv")

    review = generation[[
        "review_id", "question_id", "question", "reference_answer", "answer", "cited_ids", "prompt_ids",
    ]].copy()
    review["context_text"] = review["prompt_ids"].map(
        lambda ids: "\n\n".join(
            f"[{sid}] {corpus[sid]['section_content']}" for sid in ids
        )
    )
    review = review.drop(columns="prompt_ids").assign(
        coverage="", contradiction="", citation_support="", note="",
    )
    review.sample(frac=1, random_state=seed).to_csv(
        run_dir / "review_template.csv", index=False, encoding="utf-8-sig",
    )


def run_generation(
    corpus: Mapping[str, Mapping[str, Any]],
    generation_questions: Any,
    generation_retrieval: Any,
    config: Any,
    generator_revision: str,
    device: Any,
    dtype: Any,
    run_dir: str | Path,
) -> None:
    """Generate answers and write JSONL, summary, evidence and review CSVs.

    ``generation_questions`` is indexed by question ID and provides a
    ``reference_answer`` column. ``generation_retrieval`` contains one row per
    method/question with ``retrieved_ids`` and ``gold_ids`` sequences.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm.auto import tqdm

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.generator_id, revision=generator_revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.generator_id, revision=generator_revision, torch_dtype=dtype,
    ).to(device).eval()

    records: list[dict[str, Any]] = []
    with (run_dir / "generation_rows.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        for row in tqdm(
            generation_retrieval.itertuples(index=False),
            total=len(generation_retrieval),
            desc="Generating answers",
        ):
            selected = list(row.retrieved_ids[: config.top_k])
            input_ids, included, omitted = _make_prompt(
                row.question, selected, corpus, tokenizer, config.max_input_tokens, device,
            )
            answer, output_tokens, seconds = _answer(
                model, tokenizer, input_ids, config.max_new_tokens,
            )
            cited = list(dict.fromkeys(_CITATION_RE.findall(answer)))
            valid = [sid for sid in cited if sid in included]
            gold_ids = list(row.gold_ids)
            gold_in_top_k = bool(set(gold_ids).intersection(selected))
            gold_in_prompt = bool(set(gold_ids).intersection(included))
            review_key = f"{run_dir.name}|{row.question_id}|{row.method}"
            review_id = "A" + hashlib.sha256(review_key.encode()).hexdigest()[:12]
            record = {
                "review_id": review_id,
                "method": row.method,
                "question_id": row.question_id,
                "question": row.question,
                "reference_answer": generation_questions.at[row.question_id, "reference_answer"],
                "answer": answer,
                "gold_ids": gold_ids,
                "top_k_ids": selected,
                "prompt_ids": included,
                "omitted_ids": omitted,
                "cited_ids": cited,
                "invalid_ids": [sid for sid in cited if sid not in included],
                "valid_citations": len(valid),
                "total_citations": len(cited),
                "gold_cited": int(bool(set(gold_ids).intersection(valid))),
                "gold_in_top_k": int(gold_in_top_k),
                "gold_in_prompt": int(gold_in_prompt),
                "context_recall": len(set(gold_ids).intersection(included)) / len(set(gold_ids)),
                "abstained": int(answer.startswith(ABSTAIN_TEXT)),
                "input_tokens": input_ids.shape[-1],
                "output_tokens": output_tokens,
                "output_token_limit_reached": int(output_tokens == config.max_new_tokens),
                "generation_seconds": seconds,
            }
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

    _summary_artifacts(records, run_dir, config.seed, corpus)

    # Keep a compact human-readable counterpart to the notebook's example
    # display without adding columns to any machine-readable artifact.
    if records and len(generation_questions.index):
        example_question_id = generation_questions.index[0]
        examples = [r for r in records if r["question_id"] == example_question_id]
        if examples:
            print("คำถามตัวอย่าง:", examples[0]["question"])
            for example in examples:
                stage = (
                    "retrieval_miss" if not example["gold_in_top_k"] else
                    "context_omitted" if not example["gold_in_prompt"] else
                    "gold_in_prompt"
                )
                resolved = [
                    f"{corpus[sid]['law_name']} มาตรา {corpus[sid]['section_num']}"
                    for sid in example["cited_ids"] if sid in example["prompt_ids"]
                ]
                print(
                    f"[{example['method']}] {example['answer']}\n"
                    f"มาตราที่อ้างได้: {resolved}\n"
                    f"invalid_ids: {example['invalid_ids']} | evidence_stage: {stage}"
                )
