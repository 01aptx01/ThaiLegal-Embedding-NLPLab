"""Command line interface. Importing this module does not load ML libraries."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import RunConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thai-legal-rag", description="Run or review Thai legal RAG evaluations.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run retrieval and optionally answer generation")
    run.add_argument("--mode", choices=("pilot", "test"), default=RunConfig.mode)
    run.add_argument("--retrieval-only", action="store_true", help="skip answer generation")
    defaults = RunConfig()
    run.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default=defaults.device)
    run.add_argument("--seed", type=int, default=defaults.seed)
    run.add_argument("--pilot-questions", type=int, default=defaults.pilot_questions)
    run.add_argument("--pilot-generation-questions", type=int, default=defaults.pilot_generation_questions)
    run.add_argument("--test-generation-questions", type=int, default=defaults.test_generation_questions)
    for name in ("corpus", "pilot", "test", "bge", "generator"):
        run.add_argument(f"--{name}-revision", metavar="REF", help=f"pin {name} to a Hugging Face revision")
    run.add_argument("--bge-batch-size", type=int, default=defaults.bge_batch_size)

    review = subparsers.add_parser("review", help="summarize human review scores and refresh the ZIP")
    review.add_argument("run_dir", type=Path)
    review.add_argument("--scores", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.command == "run":
            config = RunConfig(
                mode=args.mode, retrieval_only=args.retrieval_only, output_dir=args.output_dir,
                device=args.device, seed=args.seed, pilot_questions=args.pilot_questions,
                pilot_generation_questions=args.pilot_generation_questions,
                test_generation_questions=args.test_generation_questions,
                corpus_revision=args.corpus_revision, pilot_revision=args.pilot_revision,
                test_revision=args.test_revision, bge_revision=args.bge_revision,
                generator_revision=args.generator_revision, bge_batch_size=args.bge_batch_size,
            )
            from .pipeline import run
            run_dir = run(config)
            print(f"Completed run: {run_dir}")
            print(f"Archive: {run_dir.with_suffix('.zip')}")
            return 0
        from .pipeline import review as review_run
        success = review_run(args.run_dir, args.scores)
        if success:
            print(f"Review summary updated: {args.run_dir}")
        else:
            print(f"Review summaries updated with no complete scored questions: {args.run_dir}")
        return 0
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
