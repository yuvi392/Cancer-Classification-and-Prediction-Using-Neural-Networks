"""Flask web application for the breast-tumor analysis prototype."""
import logging
import threading
import webbrowser
from flask import Flask, flash, redirect, render_template, request, url_for
from config import FIGURES_DIR
from src.data_loader import load_dataset
from src.predict import predict

app = Flask(__name__)
app.config["SECRET_KEY"] = "local-development-only"
logging.basicConfig(level=logging.INFO)

def field_key(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

@app.context_processor
def inject_fields():
    bundle = load_dataset()
    return {"feature_names": bundle.feature_names, "field_key": field_key}

@app.get("/")
def index(): return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def prediction_form():
    if request.method == "GET": return render_template("predict.html")
    bundle = load_dataset()
    try:
        values = [float(request.form[field_key(name)]) for name in bundle.feature_names]
        result = predict(values)
        return render_template("results.html", result=result)
    except (KeyError, ValueError, TypeError) as exc:
        flash(f"Please provide 30 valid finite numeric measurements. ({exc})", "error")
        return render_template("predict.html"), 400
    except FileNotFoundError as exc:
        flash(str(exc), "error")
        return render_template("predict.html"), 503

@app.get("/dashboard")
def dashboard():
    metrics_path = __import__("config").METRICS_DIR / "metrics.json"
    metrics = None
    if metrics_path.exists():
        import json; metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return render_template("dashboard.html", metrics=metrics, figures=FIGURES_DIR.exists())

@app.get("/about")
def about(): return render_template("about.html")

@app.errorhandler(404)
def not_found(_error): return render_template("404.html"), 404

def open_browser() -> None:
    """Open the local app after Flask has had time to start listening."""
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
