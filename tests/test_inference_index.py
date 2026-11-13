"""Offline smoke tests for building and querying the inference index."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from thai_legal_rag import inference_index


class TinyVectors:
    vector_size = 2

    def __contains__(self, token):
        return token in {"สัญญา", "รถ"}

    def __getitem__(self, token):
        return np.array([1., 0.], dtype=np.float32) if token == "สัญญา" else np.array([0., 1.], dtype=np.float32)


class TinyWord2Vec:
    def __init__(self, **kwargs):
        self.wv = TinyVectors()

    @property
    def vector_size(self):
        return self.wv.vector_size


class InferenceIndexTests(unittest.TestCase):
    def test_build_load_and_query_sparse_methods_offline(self):
        corpus = {
            "S00001": {"law_name": "กฎหมายหนึ่ง", "section_num": "๑", "section_content": "สัญญา รถ"},
            "S00002": {"law_name": "กฎหมายสอง", "section_num": "๒", "section_content": "รถ"},
        }
        models = types.ModuleType("gensim.models")
        models.Word2Vec = TinyWord2Vec
        gensim = types.ModuleType("gensim")
        gensim.models = models
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(inference_index, "_load_corpus", return_value=(corpus, "a" * 40)), \
                patch.object(inference_index, "_tokenize", side_effect=lambda s: s.split()), \
                patch.dict(sys.modules, {"gensim": gensim, "gensim.models": models}):
            target = Path(tmp) / "index"
            inference_index.build_index(target, include_dense=False)
            index = inference_index.InferenceIndex.load(target, device="cpu")
            self.assertEqual(list(index.methods), ["r1", "r2", "r3"])
            self.assertEqual(index.corpus_revision, "a" * 40)
            self.assertEqual(index.corpus["S00001"]["law_name"], "กฎหมายหนึ่ง")
            self.assertEqual(index.search("สัญญา", "r1", 1), ["S00001"])
            self.assertEqual(index.search("สัญญา", "r3", 1), ["S00001"])
            self.assertEqual(index.search("คำที่ไม่มี", "r1"), [])
            with self.assertRaises(ValueError):
                index.search("สัญญา", "r4")


if __name__ == "__main__":
    unittest.main()
