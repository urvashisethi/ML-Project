from flask import Flask  , render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form values
    input_data = {
        'age': int(request.form['age']),
        'sex': request.form['sex'],
        'cp': request.form['cp'],
        'trestbps': float(request.form['trestbps']),
        'chol': float(request.form['chol']),
        'fbs': request.form['fbs'],
        'restecg': request.form['restecg'],
        'thalch': float(request.form['thalch']),
        'exang': request.form['exang'],
        'oldpeak': float(request.form['oldpeak']),
        'slope': request.form['slope'],
        'ca': float(request.form['ca']),
        'thal': request.form['thal']
    }

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    result = "has Heart Disease" if prediction > 0 else "does NOT have Heart Disease"

    return f"<h3>Prediction: The patient {result}</h3><a href='/'>Try Again</a>"

if __name__ == "__main__":
    app.run(debug=True)
    
    
