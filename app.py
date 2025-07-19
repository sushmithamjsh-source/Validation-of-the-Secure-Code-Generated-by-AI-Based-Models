from flask import Flask, render_template, request
from keras.models import load_model
from scripts.ml_module import predict_vulnerability
from scripts.preprocessing import preprocess_code_sliding_window
from scripts.non_ml_module import run_bandit
from scripts.aggregator import aggregate_results_all
from scripts import myutils
import numpy as np

app = Flask(__name__)

# Load Word2Vec-based LSTM models
models = {
    "command_injection": load_model("models/Bidirectional_LSTM_model_command_injection.h5", custom_objects={'f1_loss': myutils.f1_loss, 'f1': myutils.f1}),
    "sql_injection": load_model("models/bidirectional_LSTM_model_sql.h5", custom_objects={'f1_loss': myutils.f1_loss, 'f1': myutils.f1}),
    "xss": load_model("models/Bidirectional_LSTM_model_xss.h5", custom_objects={'f1_loss': myutils.f1_loss, 'f1': myutils.f1}),
    "remote_code_execution": load_model("models/Bidirectional_LSTM_model_remote_code_execution.h5", custom_objects={'f1_loss': myutils.f1_loss, 'f1': myutils.f1}),
    "open_redirect": load_model("models/Bidirectional_LSTM_model_open_redirect.h5", custom_objects={'f1_loss': myutils.f1_loss, 'f1': myutils.f1}),
    "path_disclosure": load_model("models/Bidirectional_LSTM_model_path_disclosure.h5", custom_objects={'f1_loss': myutils.f1_loss, 'f1': myutils.f1}),
    "xsrf": load_model("models/bidirectional_LSTM_model_xsrf.h5", custom_objects={'f1_loss': myutils.f1_loss, 'f1': myutils.f1}),
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        code_text = request.form.get("code")
        if not code_text or code_text.strip() == "":
            return "No code provided."

        filepath = "temp_code.py"
        with open(filepath, "w") as f:
            f.write(code_text)

        chunks = preprocess_code_sliding_window(filepath, maxlen=200, stride=100)

        ml_results = {}
        for vuln_type, model in models.items():
            vuln_detected = False
            for chunk in chunks:
                input_chunk = chunk.reshape(1, 200, 300)  # Word2Vec input shape
                label = predict_vulnerability(model, input_chunk)
                if label == "vulnerable":
                    vuln_detected = True
                    break
            ml_results[vuln_type] = {
                "label": "vulnerable" if vuln_detected else "secure",
                "vulnerability_type": vuln_type if vuln_detected else None
            }

        non_ml_result = run_bandit(filepath)
        final_result = aggregate_results_all(ml_results, non_ml_result)
        return render_template("result.html", result=final_result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, port=5050)
