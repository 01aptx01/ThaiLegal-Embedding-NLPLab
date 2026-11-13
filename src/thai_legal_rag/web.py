"""Serve the research workspace and stream results from warm inference models."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from contextlib import asynccontextmanager, closing
from pathlib import Path
from threading import BoundedSemaphore, Event
from time import perf_counter

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1500)
    methods: list[str] = Field(min_length=1, max_length=4)
    generate: bool = True

    @field_validator("question")
    @classmethod
    def trim_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must contain text")
        return value.strip()

    @field_validator("methods")
    @classmethod
    def unique_methods(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("methods must be unique")
        return value


def _event_stream(index, generator, query, cancelled):
    for method in query.methods:
        if cancelled.is_set():
            return
        try:
            started = perf_counter()
            ranked = index.search(query.question, method, limit=10)
            retrieval_ms = (perf_counter() - started) * 1000
            evidence = []
            for statute_id in ranked:
                row = index.corpus[statute_id]
                evidence.append({
                    "id": statute_id, "law_name": row["law_name"],
                    "section_num": str(row["section_num"]), "text": row["section_content"],
                })
            yield {
                "type": "retrieval", "method": method, "evidence": evidence,
                "retrieved_ids": ranked, "retrieval_ms": round(retrieval_ms, 1),
            }
            if not query.generate:
                yield {
                    "type": "result", "method": method, "answer": None,
                    "prompt_ids": [], "omitted_ids": [], "cited_ids": [], "invalid_ids": [],
                    "input_tokens": None, "output_tokens": None, "generation_ms": None, "ttft_ms": None,
                }
                continue
            with closing(generator.events(query.question, ranked[:5], index.corpus, cancelled)) as events:
                for event in events:
                    yield {**event, "method": method}
        except Exception:
            if cancelled.is_set():
                return
            logger.exception("Inference failed for method %s", method)
            yield {"type": "error", "method": method, "message": "ประมวลผลไม่สำเร็จ ลองใหม่อีกครั้ง"}
    yield {"type": "done"}


def _read_evaluation(run_dir: Path | None) -> dict:
    if run_dir is None:
        return {"rows": [], "mode": None}
    with (run_dir / "retrieval_summary.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    manifest = run_dir / "manifest.json"
    mode = json.loads(manifest.read_text(encoding="utf-8")).get("mode") if manifest.is_file() else None
    return {"rows": rows, "mode": mode}


def create_app(
    *,
    index=None,
    generator=None,
    index_dir: Path | None = None,
    evaluation_dir: Path | None = None,
    frontend_dir: Path | None = None,
    device: str = "auto",
    generator_revision: str | None = None,
    retrieval_only: bool = False,
    max_new_tokens: int = 256,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if index is None and index_dir is not None:
            from .inference_index import InferenceIndex

            app.state.index = InferenceIndex.load(index_dir, device=device)
        if app.state.index is not None and generator is None and index_dir is not None and not retrieval_only:
            from .inference_generation import LiveGenerator

            app.state.generator = LiveGenerator.load(
                device=device, revision=generator_revision, max_new_tokens=max_new_tokens
            )
        yield

    app = FastAPI(title="Thai Legal RAG", lifespan=lifespan)
    app.state.index = index
    app.state.generator = generator
    worker_slot = BoundedSemaphore(1)

    @app.get("/api/status")
    def status(request: Request):
        live_index = request.app.state.index
        live_generator = request.app.state.generator
        return {
            "ready": live_index is not None,
            "methods": (
                [{"id": key, "label": label} for key, label in live_index.methods.items()]
                if live_index is not None else []
            ),
            "generator": live_generator.model_id if live_generator is not None else None,
            "generator_revision": getattr(live_generator, "revision", None),
            "corpus_size": len(live_index.corpus) if live_index is not None else 0,
            "corpus_revision": live_index.corpus_revision if live_index is not None else None,
        }

    @app.get("/api/evaluate")
    def evaluate():
        try:
            return _read_evaluation(evaluation_dir)
        except (OSError, ValueError):
            logger.exception("Cannot read configured benchmark results")
            raise HTTPException(503, "อ่านผล benchmark ที่ตั้งค่าไว้ไม่ได้") from None

    @app.post("/api/infer")
    def infer(query: InferenceRequest, request: Request):
        live_index = request.app.state.index
        live_generator = request.app.state.generator
        if live_index is None:
            raise HTTPException(503, "ยังไม่มีดัชนีสำหรับค้นตัวบท")
        if any(method not in live_index.methods for method in query.methods):
            raise HTTPException(422, "Retriever ที่เลือกยังไม่พร้อมใช้งาน")
        if query.generate and live_generator is None:
            raise HTTPException(409, "เซิร์ฟเวอร์นี้เปิดเฉพาะการค้นหลักฐาน")

        cancelled = Event()

        def stream():
            if not worker_slot.acquire(blocking=False):
                yield json.dumps({"type": "error", "message": "ระบบกำลังตอบคำถามอื่น ลองใหม่อีกครั้ง"}) + "\n"
                yield '{"type":"done"}\n'
                return
            try:
                with closing(_event_stream(live_index, live_generator, query, cancelled)) as events:
                    for event in events:
                        yield json.dumps(event, ensure_ascii=False) + "\n"
            finally:
                cancelled.set()
                worker_slot.release()

        iterator = stream()

        async def chunks():
            try:
                while (chunk := await anyio.to_thread.run_sync(next, iterator, None)) is not None:
                    yield chunk
            finally:
                cancelled.set()

        class InferenceResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    async with anyio.create_task_group() as group:
                        async def send_stream():
                            try:
                                await self.stream_response(send)
                            finally:
                                group.cancel_scope.cancel()

                        group.start_soon(send_stream)
                        await self.listen_for_disconnect(receive)
                        cancelled.set()
                        group.cancel_scope.cancel()
                finally:
                    cancelled.set()
                    with anyio.CancelScope(shield=True):
                        await self.body_iterator.aclose()
                        await anyio.to_thread.run_sync(iterator.close)

        return InferenceResponse(
            chunks(), media_type="application/x-ndjson",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    if frontend_dir is not None:
        assets = frontend_dir / "assets"
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/", include_in_schema=False)
        def workspace():
            return FileResponse(frontend_dir / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="thai-legal-rag-web")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-index", help="prepare the corpus and search index once")
    build.add_argument("--output", type=Path, default=Path("inference_index"))
    build.add_argument("--corpus-revision")
    build.add_argument("--bge-revision")
    build.add_argument("--batch-size", type=int, default=8)
    build.add_argument("--no-dense", action="store_true", help="build R1–R3 without BGE-M3")
    serve = commands.add_parser("serve", help="start the API and serve the compiled frontend")
    serve.add_argument("--index", type=Path, default=Path("inference_index"))
    serve.add_argument("--frontend", type=Path, default=Path("frontend/dist"))
    serve.add_argument("--evaluation-run", type=Path)
    serve.add_argument("--generator-revision")
    serve.add_argument("--max-new-tokens", type=int, default=256)
    serve.add_argument("--retrieval-only", action="store_true")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    for command in (build, serve):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "build-index":
        from .inference_index import build_index

        build_index(
            args.output, corpus_revision=args.corpus_revision, bge_revision=args.bge_revision,
            device=args.device, batch_size=args.batch_size, include_dense=not args.no_dense,
        )
        return 0

    import uvicorn

    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be between 1 and 256")

    if not (args.frontend / "index.html").is_file():
        parser.error(f"Frontend build missing at {args.frontend}. Run npm ci and npm run build in frontend/.")
    if not args.index.is_dir():
        parser.error(f"Index missing at {args.index}. Run thai-legal-rag-web build-index first.")
    uvicorn.run(
        create_app(
            index_dir=args.index, frontend_dir=args.frontend, evaluation_dir=args.evaluation_run,
            device=args.device, generator_revision=args.generator_revision,
            retrieval_only=args.retrieval_only, max_new_tokens=args.max_new_tokens,
        ),
        host=args.host, port=args.port, access_log=False,
    )
    return 0
