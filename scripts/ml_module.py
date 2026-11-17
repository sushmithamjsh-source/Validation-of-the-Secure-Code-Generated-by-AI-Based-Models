import numpy as np

DEFAULT_THRESHOLD = 0.5


def predict_vulnerability(model, input_chunk, threshold=DEFAULT_THRESHOLD):
    """Classify a single (1, maxlen, dim) chunk as 'vulnerable' or 'secure'."""
    score = float(model.predict(input_chunk, verbose=0)[0][0])
    return "vulnerable" if score > threshold else "secure"


def predict_chunks(model, chunks, threshold=DEFAULT_THRESHOLD):
    """
    Score all chunks in one batch. Returns (label, max_score).
    Code is 'vulnerable' if any chunk scores above the threshold.
    """
    if not chunks:
        return "inconclusive", None
    batch = np.stack(chunks).astype(np.float32)
    scores = model.predict(batch, verbose=0)[:, 0]
    max_score = float(scores.max())
    return ("vulnerable" if max_score > threshold else "secure"), max_score
