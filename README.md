# AI Code Validation

A Flask web app that checks Python code for security vulnerabilities by combining
**deep learning** (Word2Vec + LSTM models) with **static analysis** (Bandit).

Paste a code snippet, and the app reports:

- a prediction from four LSTM models, one per vulnerability type:
  **SQL injection**, **command injection**, **XSS** and **remote code execution**
- Bandit's rule-based findings with their severity
- an overall verdict that combines both

![Results page for a snippet containing SQL and command injection](docs/screenshot.png)

## How it works

```
code ──► strip comments ──► tokenize ──► Word2Vec (300-d) ──► sliding windows (200 tokens, stride 100)
                                                                         │
                                      ┌──────────────────────────────────┤
                                      ▼                                  ▼
                           4 × LSTM classifiers                    Bandit (static)
                                      └───────────────┬──────────────────┘
                                                      ▼
                                          aggregated verdict
```

1. **Preprocessing** ([scripts/preprocessing.py](scripts/preprocessing.py)): the code is tokenized
   (identifiers *and* operators/punctuation are kept as tokens), each token is embedded with a
   Word2Vec model trained on Python source, and the sequence is cut into overlapping windows of
   200 tokens.
2. **ML detection** ([scripts/ml_module.py](scripts/ml_module.py)): each LSTM model scores every
   window; the code is flagged for that vulnerability type if any window scores above 0.5.
3. **Static analysis** ([scripts/non_ml_module.py](scripts/non_ml_module.py)): Bandit runs on the
   same code.
4. **Aggregation** ([scripts/aggregator.py](scripts/aggregator.py)): the overall status is
   *vulnerable* if any model flags the code or Bandit reports a MEDIUM/HIGH severity issue.

## Model performance

<!-- Fill in with your evaluation results on the held-out test set. -->

| Model | Precision | Recall | F1 |
|---|---|---|---|
| SQL injection | – | – | – |
| Command injection | – | – | – |
| XSS | – | – | – |
| Remote code execution | – | – | – |

## Getting started

Requires Python 3.9–3.11 (TensorFlow 2.15).

```bash
git clone <this-repo-url>
cd AI_Code_Validation_Project_w2v
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5050.

### Running the tests

```bash
python -m pytest
```

## Project structure

```
app.py                  Flask app (routes, model loading)
scripts/
  preprocessing.py      tokenization, Word2Vec embedding, sliding windows
  ml_module.py          LSTM inference
  non_ml_module.py      Bandit wrapper
  aggregator.py         combines ML + Bandit results
  myutils.py            tokenizer and F1 metric helpers
models/                 trained Word2Vec and LSTM models
templates/              HTML pages
tests/                  pytest tests
```

## Limitations

- Python code only.
- Comment stripping is line-based, so a `#` inside a string literal is treated as a comment.
- Models judge token patterns, not data flow, so they can miss vulnerabilities or raise false
  positives; treat results as a review aid, not a guarantee.

## Credits

The LSTM and Word2Vec models in this repository were trained by me. The approach (Word2Vec
embeddings of code tokens fed into per-vulnerability LSTM classifiers) and the tokenization code
in [scripts/myutils.py](scripts/myutils.py) are adapted from **VUDENC**:

> L. Wartschinski, Y. Noller, T. Vogel, T. Kehrer, L. Grunske.
> *VUDENC: Vulnerability Detection with Deep Learning on a Natural Codebase for Python.*
> Information and Software Technology, 2022.
> https://github.com/LauraWartschinski/VulnerabilityDetection
