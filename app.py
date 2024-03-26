from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)

# Load the trained model
rf_classifier = joblib.load('rf_model.joblib')
vectorizer = CountVectorizer()

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_counts = vectorizer.transform([text])
    prediction = rf_classifier.predict(text_counts)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
