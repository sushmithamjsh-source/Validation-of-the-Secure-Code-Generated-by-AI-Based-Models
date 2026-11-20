import numpy as np

from scripts.ml_module import predict_chunks


class FakeModel:
    """Stands in for a Keras model: returns fixed scores, one per chunk."""

    def __init__(self, scores):
        self.scores = scores

    def predict(self, batch, verbose=0):
        assert batch.shape == (len(self.scores), 100, 300)
        return np.array(self.scores, dtype=np.float32).reshape(-1, 1)


def chunks(n):
    return [np.zeros((100, 300), dtype=np.float32) for _ in range(n)]


def test_any_chunk_above_threshold_is_vulnerable():
    label, score = predict_chunks(FakeModel([0.1, 0.8, 0.2]), chunks(3))
    assert label == "vulnerable"
    assert score == np.float32(0.8)


def test_all_chunks_below_threshold_is_secure():
    label, _ = predict_chunks(FakeModel([0.1, 0.3]), chunks(2))
    assert label == "secure"


def test_no_chunks_is_inconclusive():
    assert predict_chunks(FakeModel([]), []) == ("inconclusive", None)
