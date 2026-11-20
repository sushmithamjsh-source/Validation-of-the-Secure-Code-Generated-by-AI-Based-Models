"""
Train one LSTM vulnerability classifier on VUDENC-style commit data.

Usage:
    python -m training.train sql --data-dir E:/lul/VulnerabilityDetection/Code/data

Fixes compared with the original VUDENC makemodel.py:
  * Inputs are kept as float32. The original called pad_sequences with its default
    dtype (int32), which truncated every Word2Vec value in (-1, 1) to 0, so the original
    models were trained on ~all-zero input.
  * Input length is 100 tokens (99% of samples fit), which halves memory use.
  * Train/validation/test are split by repository (fixed seed) instead of per sample, so
    near-duplicate overlapping samples can't leak from training into the test set.
  * Metrics are written to training/output/metrics_<mode>.json instead of only printed.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.utils import Sequence

from scripts import myutils
from training.vudenc_data import findposition, findpositions, getblocks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W2V_PATH = os.path.join(ROOT, "models", "word2vec_withString10-200-300.model")
OUT_DIR = os.path.join(ROOT, "training", "output")

STEP = 5            # step length when sliding over a file (chars)
FULLLENGTH = 200    # context length of one sample (chars)
MAX_LEN = 100       # tokens per sample after padding (model input length); p99 of sample length is ~100
# Samples per split after the repository split (random subsample) — keeps memory and
# epoch time manageable on a laptop
MAX_SAMPLES = {"train": 10000, "validation": 2500, "test": 2500}
SEED = 42


def load_blocks(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    blocks = []
    for repo_name, repo in data.items():
        for commit in repo.values():
            for fdata in commit.get("files", {}).values():
                if "source" not in fdata:
                    continue
                source = fdata["source"]
                badparts = [bad for change in fdata["changes"] for bad in change["badparts"]
                            if -1 not in findposition(bad, source)]
                if not badparts:
                    continue
                for code, label in getblocks(source, findpositions(badparts, source), STEP, FULLLENGTH):
                    blocks.append((code, label, repo_name))
    return blocks


def split_by_repo(repos, rng):
    """
    70/15/15 split where every repository lands in exactly one set. Neighbouring samples
    from the same file overlap almost entirely, so a per-sample random split would leak
    near-duplicates into the test set and inflate the scores.
    """
    order = sorted(set(repos))
    rng.shuffle(order)
    sizes = {r: 0 for r in order}
    for r in repos:
        sizes[r] += 1
    # Fill test, then validation, until each has ~15% of the samples; the rest is train.
    # Filling the small splits first guarantees each gets at least one repository.
    assignment, filled = {}, {"test": 0, "validation": 0}
    for r in order:
        target = next((s for s in ("test", "validation") if filled[s] < 0.15 * len(repos)), "train")
        assignment[r] = target
        if target in filled:
            filled[target] += sizes[r]
    splits = {"train": [], "validation": [], "test": []}
    for i, r in enumerate(repos):
        splits[assignment[r]].append(i)
    return splits


def token_ids(code, wv):
    return [wv.key_to_index[t] for t in myutils.getTokens(code) if t != " " and t in wv]


def to_array(id_lists, wv):
    """
    Embed and pre-pad (keep the last MAX_LEN tokens), written straight into one float32 array.
    Equivalent to pad_sequences(..., dtype="float32") but without an intermediate copy.
    """
    xs = np.zeros((len(id_lists), MAX_LEN, wv.vector_size), dtype=np.float32)
    for i, ids in enumerate(id_lists):
        ids = ids[-MAX_LEN:]
        if ids:
            xs[i, MAX_LEN - len(ids):] = wv.vectors[ids]
    return xs


class Batches(Sequence):
    """Builds float32 batches on the fly so the full (n, MAX_LEN, 300) array is never in memory."""

    def __init__(self, id_lists, y, wv, batch_size=128, shuffle=False):
        super().__init__()
        self.id_lists, self.y, self.wv = id_lists, np.asarray(y), wv
        self.batch_size, self.shuffle = batch_size, shuffle
        self.order = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, i):
        sel = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        return to_array([self.id_lists[k] for k in sel], self.wv), self.y[sel].astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


def metrics(y_true, probs, threshold=0.5):
    y_pred = (probs > threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": round((tp + tn) / len(y_true), 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y_true)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="vulnerability type, e.g. sql, xss, command_injection")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    blocks = load_blocks(os.path.join(args.data_dir, "plain_" + args.mode))
    print(f"[{args.mode}] {len(blocks)} samples extracted ({time.time() - t0:.0f}s)", flush=True)

    wv = Word2Vec.load(W2V_PATH).wv
    X = [token_ids(code, wv) for code, _, _ in blocks]
    y = np.array([label for _, label, _ in blocks])
    rng = random.Random(SEED)
    splits = split_by_repo([repo for _, _, repo in blocks], rng)
    repos_per_split = {name: len({blocks[i][2] for i in keys}) for name, keys in splits.items()}
    for name, keys in splits.items():
        if len(keys) > MAX_SAMPLES[name]:
            splits[name] = rng.sample(keys, MAX_SAMPLES[name])
    n_repos = len({repo for _, _, repo in blocks})
    del blocks
    token_lengths = [len(x) for x in X]
    print(f"[{args.mode}] {len(y)} samples from {n_repos} repositories; "
          f"repositories per split: {repos_per_split}", flush=True)

    batches = {name: Batches([X[k] for k in keys], y[keys], wv, batch_size=args.batch_size,
                             shuffle=(name == "train"))
               for name, keys in splits.items()}
    y_train = batches["train"].y
    print(f"[{args.mode}] train={len(y_train)} val={len(splits['validation'])} "
          f"test={len(splits['test'])} vulnerable={y.mean():.1%} "
          f"median_tokens={np.median(token_lengths):.0f}", flush=True)

    # Balanced class weights for the imbalanced data
    counts = np.bincount(y_train, minlength=2)
    class_weight = {c: len(y_train) / (2 * counts[c]) for c in (0, 1) if counts[c]}

    model = Sequential([
        LSTM(100, dropout=0.2, recurrent_dropout=0.2, input_shape=(MAX_LEN, wv.vector_size)),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(batches["train"], validation_data=batches["validation"], epochs=args.epochs,
              class_weight=class_weight, verbose=2,
              callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)])

    result = {"mode": args.mode, "seed": SEED, "max_len": MAX_LEN, "split": "by repository",
              "repositories": n_repos, "repositories_per_split": repos_per_split,
              "token_length": {"median": float(np.median(token_lengths)),
                               "p90": float(np.percentile(token_lengths, 90))}}
    for name, b in batches.items():
        b.shuffle = False
        b.order = np.arange(len(b.y))
        result[name] = metrics(b.y, model.predict(b, verbose=0)[:, 0])
        print(f"[{args.mode}] {name}: {result[name]}", flush=True)

    model.save(os.path.join(OUT_DIR, f"LSTM_model_{args.mode}.h5"))
    with open(os.path.join(OUT_DIR, f"metrics_{args.mode}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{args.mode}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
