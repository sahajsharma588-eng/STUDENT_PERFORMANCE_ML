from flask import Flask,render_template, request

import joblib

app = Flask(__name__,template_folder="templates")

model = joblib.load("student_performance_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_score = float(request.form["previous_score"])
    assignments = int(request.form["asignment_completed"])
    input_data = [[study_hours,attendance,previous_score,assignments]]
    prediction = model.predict(input_data)
    return f"<h1>predicted final score: {prediction[0]:.2f}</h1>"
if __name__=="__main__":
    app.run(debug=True)
