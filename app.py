from pathlib import Path
import pickle
import warnings

from flask import Flask, render_template, request

from model import train_and_save_model

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"


def load_or_create_model():
    if not MODEL_PATH.exists():
        return train_and_save_model()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with MODEL_PATH.open("rb") as model_file:
                return pickle.load(model_file)
    except (Warning, pickle.PickleError, EOFError, AttributeError, ValueError):
        # Rebuild the model if the serialized file is incompatible or corrupted.
        return train_and_save_model()


model = load_or_create_model()

def get_risk(pred):
    if pred < 50:
        return "high", "High Risk 🔴"
    elif pred < 70:
        return "medium", "Medium Risk 🟡"
    else:
        return "low", "Low Risk 🟢"

def get_recommendation(att, study):
    rec = []
    if att < 75:
        rec.append("Increase attendance")
    if study < 3:
        rec.append("Study at least 3 hrs/day")
    return rec if rec else ["Keep it up!"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['attendance']),
            float(request.form['internal']),
            float(request.form['gpa']),
            float(request.form['study'])
        ]
    except (KeyError, TypeError, ValueError):
        return render_template(
            "index.html",
            error="Please enter valid numeric values for all fields.",
        )

    pred = model.predict([data])[0]
    risk_class, risk = get_risk(pred)
    rec = get_recommendation(data[0], data[3])

    return render_template("index.html",
                           prediction=round(pred,2),
                           risk=risk,
                           risk_class=risk_class,
                           rec=rec)

if __name__ == "__main__":
    app.run(debug=True)