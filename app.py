import os
import tempfile

from flask import Flask, render_template, request
from keras.models import load_model

from scripts.ml_module import predict_chunks
from scripts.preprocessing import preprocess_code_sliding_window
from scripts.non_ml_module import run_bandit
from scripts.aggregator import aggregate_results_all
from scripts import myutils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB upload limit

CUSTOM_OBJECTS = {"f1_loss": myutils.f1_loss, "f1": myutils.f1}


def _load(name):
    # compile=False: we only run inference, so the training loss/metrics aren't needed
    return load_model(os.path.join(MODELS_DIR, name), custom_objects=CUSTOM_OBJECTS, compile=False)


# Load Word2Vec-based LSTM models
models = {
    "command_injection": _load("LSTM_model_command_injection.h5"),
    "sql_injection": _load("LSTM_model_sql.h5"),
    "xss": _load("LSTM_model_xss.h5"),
    "remote_code_execution": _load("LSTM_model_remote_code_execution.h5"),
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        code_text = request.form.get("code")
        if not code_text or code_text.strip() == "":
            return "No code provided."

        chunks = preprocess_code_sliding_window(code_text, maxlen=200, stride=100)

        ml_results = {}
        for vuln_type, model in models.items():
            label, score = predict_chunks(model, chunks)
            ml_results[vuln_type] = {
                "label": label,
                "score": score,
                "vulnerability_type": vuln_type if label == "vulnerable" else None,
            }

        # Unique temp file per request so concurrent users don't overwrite each other
        fd, filepath = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code_text)
            non_ml_result = run_bandit(filepath)
        finally:
            os.remove(filepath)

        final_result = aggregate_results_all(ml_results, non_ml_result)
        return render_template("result.html", result=final_result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False, port=5050)
