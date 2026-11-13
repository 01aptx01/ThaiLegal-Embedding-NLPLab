import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from thai_legal_rag.retrieval import SUMMARY_COLS, _summary, rank_exact, score_ranking


class RetrievalTests(unittest.TestCase):
    def test_exact_cosine_scores_rank_hand_calculated(self):
        # q=(1,0); cosine similarities are 1, 0, and 1/sqrt(2).
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        docs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        norms = np.linalg.norm(docs, axis=1, keepdims=True)
        docs = docs / norms
        self.assertEqual(rank_exact(q, docs, ["A", "B", "C"], 3), [["A", "C", "B"]])

    def test_exact_rank_preserves_input_order_for_ties(self):
        q = csr_matrix([[1.0, 0.0]], dtype=np.float32)
        docs = csr_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.assertEqual(rank_exact(q, docs, ["first", "second", "third"], 3),
                         [["first", "second", "third"]])

    def test_zero_query_returns_no_retrieval(self):
        q = np.zeros((1, 2), dtype=np.float32)
        docs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.assertEqual(rank_exact(q, docs, ["A", "B"], 2), [[]])
        self.assertEqual(score_ranking(["A"], [], 5)["ndcg"], 0)

    def test_empty_summary_matches_nonempty_csv_schema(self):
        empty = _summary(pd.DataFrame())
        rows = pd.DataFrame({
            "method": ["R1 Binary BoW"], "recall_at_1": [1.0], "recall_at_5": [1.0],
            "recall_at_10": [1.0], "hit_at_5": [1], "ndcg_at_5": [1.0],
            "ndcg_at_10": [1.0], "rr_at_10": [1.0], "empty_retrieval": [0],
        })
        populated = _summary(rows)
        self.assertEqual(list(empty.columns), list(populated.columns))
        self.assertEqual(list(empty.columns), SUMMARY_COLS)
        self.assertEqual(empty.index.name, populated.index.name)
        self.assertEqual(empty.index.name, "method")

    def test_score_ranking_requires_positive_k(self):
        with self.assertRaises(ValueError):
            score_ranking(["A"], ["A"], 0)
