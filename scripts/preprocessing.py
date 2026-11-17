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


def preprocess_code_sliding_window(code, maxlen=200, stride=100):
    """
    Turn source code into a list of (maxlen, embedding_dim) arrays.
    Like training, out-of-vocabulary tokens are skipped and short
    sequences are zero-padded at the front ('pre' padding).
    """
    vectors = [w2v_model.wv[t] for t in tokenize_code(code) if t in w2v_model.wv]
    if not vectors:
        return []
    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(vectors)

    starts = list(range(0, max(n - maxlen, 0) + 1, stride))
    if starts[-1] + maxlen < n:  # make sure the tail of the code is covered
        starts.append(n - maxlen)

    chunks = []
    for start in starts:
        window = vectors[start:start + maxlen]
        chunk = np.zeros((maxlen, embedding_dim), dtype=np.float32)
        chunk[maxlen - len(window):] = window
        chunks.append(chunk)
    return chunks
