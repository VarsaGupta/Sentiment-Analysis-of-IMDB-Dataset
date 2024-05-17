from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Set the path to the model file
model_path = "C:\\Users\\varsha gupta\\Downloads\\IBMD-SentimentAnalysis\\Sentiment_Analysis_Case_Study-main\\models\\model\\model_2.pkl"

# Load the classifier
classifier = joblib.load(model_path)

def predictfunc(review):    
    prediction = classifier.predict(review)
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    return prediction[0], sentiment

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        content = request.form['review']
        review = pd.Series([content])
        prediction, sentiment = predictfunc(review)      
        return render_template("predict.html", pred=prediction, sent=sentiment)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
