"""Exact-cosine retrieval baselines used by the Thai legal RAG experiment."""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import search_text

logger = logging.getLogger(__name__)
METHODS = ("R1 Binary BoW", "R2 TF-IDF", "R3 Word2Vec", "R4 BGE-M3")
METRIC_COLS = ["recall_at_1", "recall_at_5", "recall_at_10", "hit_at_5", "ndcg_at_5", "ndcg_at_10", "rr_at_10"]
SUMMARY_COLS = ["recall_at_1", "recall_at_5", "recall_at_10", "hit_at_5", "ndcg_at_5", "ndcg_at_10", "mrr_at_10", "questions", "empty_queries"]


def score_ranking(gold, ranked, k):
    if k <= 0:
        raise ValueError("k must be positive")
    gold = set(gold)
    if not gold:
        raise ValueError("gold must not be empty")
    ranked = list(dict.fromkeys(ranked))[:k]
    rel = [int(sid in gold) for sid in ranked]
    recall = sum(rel) / len(gold)
    hit = int(any(rel))
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gold))))
    first = next((i for i, r in enumerate(rel, 1) if r), None)
    return {"recall": recall, "hit": hit, "ndcg": dcg / ideal, "rr": 1 / first if first else 0.0}


def rank_exact(query_vectors, statute_vectors, statute_ids, rank_k):
    """Stable descending cosine order; zero-vector queries have no results."""
    all_ranks = []
    for start in range(0, query_vectors.shape[0], 64):
        scores = query_vectors[start:start + 64] @ statute_vectors.T
        if issparse(scores):
            scores = scores.toarray()
        for offset, row in enumerate(scores):
            q = query_vectors[start + offset]
            is_zero = q.nnz == 0 if issparse(q) else not np.any(q)
            if is_zero:
                all_ranks.append([])
                continue
            order = np.argsort(-row, kind="stable")[:rank_k]
            all_ranks.append([statute_ids[i] for i in order])
    return all_ranks


def _tokenize(text):
    from pythainlp.tokenize import word_tokenize
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def _word2vec_average(token_rows, model):
    vectors = np.zeros((len(token_rows), model.vector_size), dtype=np.float32)
    for i, tokens in enumerate(token_rows):
        known = [model.wv[token] for token in tokens if token in model.wv]
        if known:
            vectors[i] = np.mean(known, axis=0)
    return normalize(vectors)


def _summary(rows):
    if rows.empty:
        result = pd.DataFrame(columns=SUMMARY_COLS)
        result.index.name = "method"
        return result
    result = rows.groupby("method", sort=False)[METRIC_COLS].mean().rename(columns={"rr_at_10": "mrr_at_10"})
    result["questions"] = rows.groupby("method", sort=False).size()
    result["empty_queries"] = rows.groupby("method", sort=False).empty_retrieval.sum()
    result = result[SUMMARY_COLS]
    result.index.name = "method"
    return result


def _retrieval_rows(ranks, questions):
    rows = []
    for method in METHODS:
        ranked_lists = ranks[method]
        for question, ranked in zip(questions.itertuples(index=False), ranked_lists, strict=True):
            rank = next((i for i, sid in enumerate(ranked, 1) if sid in question.gold_ids), 0)
            row = {
                "method": method,
                "question_id": question.question_id,
                "question": question.question,
                "gold_ids": question.gold_ids,
                "retrieved_ids": ranked,
                "first_gold_rank": rank,
                "empty_retrieval": int(not ranked),
            }
            for k in (1, 5, 10):
                result = score_ranking(question.gold_ids, ranked, k)
                for key in ("recall", "hit", "ndcg", "rr"):
                    row[f"{key}_at_{k}"] = result[key]
            rows.append(row)
    return pd.DataFrame(rows)


def run_retrieval(corpus, questions, config, bge_revision, device, dtype, run_dir):
    """Run the four source-notebook rankers and persist audit/results tables."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if questions.empty:
        raise ValueError("retrieval requires at least one question")
    statute_ids = list(corpus)
    statute_texts = [search_text(corpus[sid]["section_content"]) for sid in statute_ids]
    query_texts = [search_text(question) for question in questions["question"]]

    # Audit untruncated BGE token lengths against the exact pinned model revision.
    from transformers import AutoTokenizer
    length_tokenizer = AutoTokenizer.from_pretrained(config.bge_id, revision=bge_revision)
    length_rows = []
    for sid, text in zip(statute_ids, statute_texts):
        length = len(length_tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        length_rows.append({"statute_id": sid, "chars": len(text), "bge_tokens": length,
                            "truncated_for_bge": length > config.bge_max_tokens})
    length_df = pd.DataFrame(length_rows, columns=["statute_id", "chars", "bge_tokens", "truncated_for_bge"])
    length_df.to_csv(run_dir / "corpus_lengths.csv", index=False)
    del length_tokenizer

    start = time.perf_counter()
    statute_tokens = [_tokenize(text) for text in statute_texts]
    query_tokens = [_tokenize(text) for text in query_texts]
    shared_tokenize_s = time.perf_counter() - start

    ranks, costs = {}, {}
    sparse_config = dict(analyzer=lambda tokens: tokens, dtype=np.float32)
    vectorizers = (
        CountVectorizer(binary=True, **sparse_config),
        TfidfVectorizer(**sparse_config),
    )
    for method, vectorizer in zip(METHODS[:2], vectorizers, strict=True):
        start = time.perf_counter()
        statute_vectors = normalize(vectorizer.fit_transform(statute_tokens))
        zero_docs = int((np.asarray(statute_vectors.getnnz(axis=1)) == 0).sum())
        index_s = time.perf_counter() - start
        start = time.perf_counter()
        query_vectors = normalize(vectorizer.transform(query_tokens))
        ranks[method] = rank_exact(query_vectors, statute_vectors, statute_ids, config.rank_k)
        costs[method] = (index_s, 1000 * (time.perf_counter() - start) / len(questions))
        logger.info("%s zero corpus vectors: %d", method, zero_docs)

    from gensim.models import Word2Vec
    start = time.perf_counter()
    w2v = Word2Vec(sentences=statute_tokens, sg=1, vector_size=150, window=5,
                   min_count=2, epochs=10, workers=1, seed=config.seed)
    statute_vectors = _word2vec_average(statute_tokens, w2v)
    zero_docs = int((np.linalg.norm(statute_vectors, axis=1) == 0).sum())
    index_s = time.perf_counter() - start
    start = time.perf_counter()
    query_vectors = _word2vec_average(query_tokens, w2v)
    ranks[METHODS[2]] = rank_exact(query_vectors, statute_vectors, statute_ids, config.rank_k)
    costs[METHODS[2]] = (index_s, 1000 * (time.perf_counter() - start) / len(questions))
    logger.info("%s zero corpus vectors: %d", METHODS[2], zero_docs)
    del w2v, statute_vectors, query_vectors

    from sentence_transformers import SentenceTransformer
    import torch
    encoder = SentenceTransformer(config.bge_id, revision=bge_revision, device=device,
                                  model_kwargs={"torch_dtype": dtype})
    encoder.max_seq_length = config.bge_max_tokens
    encode_options = dict(batch_size=config.bge_batch_size, show_progress_bar=True)
    start = time.perf_counter()
    statute_vectors = normalize(encoder.encode(statute_texts, **encode_options).astype(np.float32))
    zero_docs = int((np.linalg.norm(statute_vectors, axis=1) == 0).sum())
    index_s = time.perf_counter() - start
    start = time.perf_counter()
    query_vectors = normalize(encoder.encode(query_texts, **encode_options).astype(np.float32))
    ranks[METHODS[3]] = rank_exact(query_vectors, statute_vectors, statute_ids, config.rank_k)
    costs[METHODS[3]] = (index_s, 1000 * (time.perf_counter() - start) / len(questions))
    logger.info("%s zero corpus vectors: %d", METHODS[3], zero_docs)
    del encoder, statute_vectors, query_vectors
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    retrieval = _retrieval_rows(ranks, questions)
    retrieval.to_json(run_dir / "retrieval_rows.jsonl", orient="records", lines=True, force_ascii=False)
    summary = _summary(retrieval)
    cost_table = pd.DataFrame.from_dict(costs, orient="index", columns=["index_seconds", "query_ms_per_question"])
    summary.join(cost_table).to_csv(run_dir / "retrieval_summary.csv")
    logger.info("Corpus: %d | questions: %d | empty retrieval rows: %d",
                len(statute_ids), len(questions), int(retrieval.empty_retrieval.sum()))
    logger.info("Shared R1–R3 Thai tokenization: %.1fs | BGE length exceedances: %d/%d",
                shared_tokenize_s, int(length_df.truncated_for_bge.sum()), len(length_df))
    return retrieval, shared_tokenize_s


def write_subset_summary(retrieval, generation_questions, run_dir):
    """Return generation-question rows and save the notebook's subset summary."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    generation_retrieval = retrieval.loc[retrieval.question_id.isin(generation_questions.index)].copy()
    _summary(generation_retrieval).to_csv(run_dir / "retrieval_on_generation_subset.csv")
    return generation_retrieval
