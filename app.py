
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
with open("voting_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            # Convert SMS text into model features
            message_vector = vectorizer.transform([message])

            # Predict
            result = model.predict(message_vector)[0]

            prediction = "Spam" if result == 1 else "Ham"

    return render_template(
        "index.html",
        prediction=prediction,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)

