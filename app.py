from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Get the project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(BASE_DIR, "student_performance_model.pkl")
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_score = float(request.form["previous_score"])
    assignments = int(request.form["assignments"])

    # Data for the ML model
    input_data = [[
        study_hours,
        attendance,
        previous_score,
        assignments
    ]]

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Open result page
    return render_template(
        "result.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)