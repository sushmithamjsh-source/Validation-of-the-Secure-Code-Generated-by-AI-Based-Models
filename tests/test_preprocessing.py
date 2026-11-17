import numpy as np

from scripts import myutils
from scripts.preprocessing import embedding_dim, preprocess_code_sliding_window, tokenize_code


def test_get_tokens_keeps_separators():
    assert myutils.getTokens("os.system(cmd) ") == ["os", ".", "system", "(", "cmd", ")", " "]


def test_tokenize_strips_comments_and_whitespace():
    tokens = tokenize_code("x = 1  # secret\n")
    assert "secret" not in tokens
    assert " " not in tokens


def test_short_code_gives_one_front_padded_chunk():
    chunks = preprocess_code_sliding_window("import os\nos.system(cmd)\n")
    assert len(chunks) == 1
    assert chunks[0].shape == (200, embedding_dim)
    assert not chunks[0][0].any()   # padding is at the front
    assert chunks[0][-1].any()      # real tokens are at the end


def test_long_code_is_split_into_overlapping_windows():
    code = "import os\nos.system(cmd)\n" * 100
    chunks = preprocess_code_sliding_window(code, maxlen=200, stride=100)
    assert len(chunks) > 1
    assert all(c.shape == (200, embedding_dim) for c in chunks)


def test_code_with_no_known_tokens_gives_no_chunks():
    assert preprocess_code_sliding_window("   \n# only a comment\n") == []
