import os

import numpy as np
from gensim.models import Word2Vec

from scripts import myutils

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W2V_PATH = os.path.join(BASE_DIR, "models", "word2vec_withString10-200-300.model")

# Load Word2Vec model
w2v_model = Word2Vec.load(W2V_PATH)
embedding_dim = w2v_model.vector_size  # should be 300


def tokenize_code(code):
    """
    Tokenize the same way the models were trained (myutils.getTokens):
    comments stripped, punctuation/operators kept as tokens, whitespace dropped.
    """
    text = myutils.stripComments(code).replace("\n", " ")
    # getTokens only emits a token when it hits a separator, so make sure the last one is flushed
    tokens = myutils.getTokens(text + " ")
    return [t for t in tokens if t.strip()]


def preprocess_code_sliding_window(code, window=50, stride=25, maxlen=100):
    """
    Turn source code into a list of (maxlen, embedding_dim) arrays.

    Mirrors training (training/train.py): each sample there is ~200 characters of code
    (median ~50 tokens) zero-padded at the front to `maxlen` tokens, with
    out-of-vocabulary tokens skipped. So the code is cut into overlapping windows
    of `window` tokens, each front-padded to `maxlen`.
    """
    vectors = [w2v_model.wv[t] for t in tokenize_code(code) if t in w2v_model.wv]
    if not vectors:
        return []
    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(vectors)
    window = min(window, maxlen)

    starts = list(range(0, max(n - window, 0) + 1, stride))
    if starts[-1] + window < n:  # make sure the tail of the code is covered
        starts.append(n - window)

    chunks = []
    for start in starts:
        window_vecs = vectors[start:start + window]
        chunk = np.zeros((maxlen, embedding_dim), dtype=np.float32)
        chunk[maxlen - len(window_vecs):] = window_vecs
        chunks.append(chunk)
    return chunks
