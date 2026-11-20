"""
Evaluate two sets of models on the same repository-held-out test sets.

Usage:
    python -m training.compare_models --data-dir E:/lul/VulnerabilityDetection/Code/data \
        --a models --b training/output
"""
import argparse
import gc
import os
import random

import numpy as np
from gensim.models import Word2Vec
from keras.models import load_model

import training.train as T
from scripts import myutils

MODES = ["xss", "sql", "command_injection", "remote_code_execution"]


def test_set(mode, data_dir, wv):
    """Rebuild exactly the test split + subsample used in training/train.py."""
    blocks = T.load_blocks(os.path.join(data_dir, "plain_" + mode))
    rng = random.Random(T.SEED)
    splits = T.split_by_repo([r for _, _, r in blocks], rng)
    for name, keys in splits.items():
        if len(keys) > T.MAX_SAMPLES[name]:
            splits[name] = rng.sample(keys, T.MAX_SAMPLES[name])
    test = splits["test"]
    X = T.to_array([T.token_ids(blocks[i][0], wv) for i in test], wv)
    y = np.array([blocks[i][1] for i in test])
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--a", default="models", help="first model directory")
    p.add_argument("--b", default="training/output", help="second model directory")
    args = p.parse_args()

    wv = Word2Vec.load(T.W2V_PATH).wv
    custom = {"f1_loss": myutils.f1_loss, "f1": myutils.f1}
    for mode in MODES:
        X, y = test_set(mode, args.data_dir, wv)
        print(f"{mode}: {len(y)} test samples, {int(y.sum())} vulnerable "
              f"(flag-everything F1 = {2 * y.mean() / (1 + y.mean()):.2f})")
        for d in (args.a, args.b):
            model = load_model(os.path.join(T.ROOT, d, f"LSTM_model_{mode}.h5"),
                               custom_objects=custom, compile=False)
            r = T.metrics(y, model.predict(X, verbose=0, batch_size=128)[:, 0])
            print(f"  {d:16s} precision={r['precision']:.2f} recall={r['recall']:.2f} f1={r['f1']:.2f}")
            del model
            gc.collect()
        del X, y
        gc.collect()


if __name__ == "__main__":
    main()
