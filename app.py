
import os
import re
import pickle
import numpy as np
import nltk
import tensorflow as tf

from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Reduce TensorFlow log messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# Get the directory containing this Python file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Download stopwords if they are not available
nltk.download("stopwords", quiet=True)

# Load the trained vectorizer
with open(os.path.join(BASE_DIR, "vectorizer.pickle"), "rb") as f:
    vect = pickle.load(f)

# Load the trained TensorFlow model
model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "model.h5")
)

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def preprocess(text, stem=False):

    text = re.sub(
        r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+",
        " ",
        str(text).lower()
    ).strip()

    tokens = []

    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)

    return " ".join(tokens)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get review text from the HTML form
        review = request.form.get("review", "").strip()

        # Validate input
        if not review:
            return render_template(
                "index.html",
                error="Please enter a review before submitting.",
                anchor="services"
            )

        # Preprocess the review
        processed_review = preprocess(review)

        if not processed_review:
            return render_template(
                "index.html",
                error="Please enter a valid review.",
                anchor="services"
            )

        # Transform text using the trained vectorizer
        vectorized_text = vect.transform(
            [processed_review]
        ).toarray()

        # Reshape according to the model's expected input
        model_input = vectorized_text.reshape(1, 1, 2500)

        # Predict sentiment
        output = model.predict(model_input, verbose=0)

        prediction_score = float(np.asarray(output).reshape(-1)[0])

        # Interpret binary sentiment output
        sentiment = (
            "Positive" if prediction_score >= 0.5 else "Negative"
        )

        return render_template(
            "index.html",
            prediction_text=sentiment,
            prediction_score=round(prediction_score * 100, 2),
            anchor="services"
        )

    except Exception as e:
        print("Prediction error:", str(e))

        return render_template(
            "index.html",
            error="Unable to process the review. Please try again.",
            anchor="services"
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 3000)),
        debug=False
    )
