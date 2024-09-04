import numpy as np
import pickle
import string
import re
import nltk
from nltk.corpus import stopwords

import os
import json
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load models and other setup code...
# function to remove username from text
def username_remover(input_txt, username):
    """removes the username handle from the data"""
    
    r = re.findall(username, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

# function to remove punctuations and stopwords
def cleaner(comment):
    
    """function to perform cleaning of the text data. For any comment
    input the function cleans the punctuations and stopwords."""
    
    punctuation_removed = [char for char in comment if char not in string.punctuation]
    punctuation_removed_join = ''.join(punctuation_removed)
    stops_removed = [word for word in punctuation_removed_join.split() if word.lower() not in stopwords.words('english')]
    
    return stops_removed

# load count vectorizer from pickle file
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))

# load NB classifier from pickle file
nb_model = pickle.load(open("model.pkl", 'rb'))

# function to predict the sentiment given text as input
def predict_sentiment(text):
    username_removed_text = username_remover(text, "@[\w]*")
    cleaned_text = cleaner(username_removed_text)

    vectorized_text = vectorizer.transform([" ".join(cleaned_text)])
    prediction = nb_model.predict(vectorized_text)[0]
    # prediction 0 is positive and prediction 1 is negative
    if prediction == 0:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return sentiment


# (Keep your existing setup code here)

@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint for SageMaker"""
    return jsonify({"status": "ok"})

@app.route('/invocations', methods=['POST'])
def predict():
    """Prediction endpoint for SageMaker"""
    if request.content_type != 'application/json':
        return jsonify({"error": "This predictor only supports JSON data"}), 415
    
    try:
        # Parse the JSON request body
        data = json.loads(request.data.decode('utf-8'))
        text = data.get("text")
        
        if not text:
            return jsonify({"error": "Missing 'text' field in request body"}), 400
        
        # Use the predict_sentiment function to predict the sentiment
        sentiment = predict_sentiment(text)
        
        # Return the sentiment as a response
        return jsonify({"sentiment": sentiment})
    
    except Exception as e:
        # If any error occurs, return a 500 error
        return jsonify({"error": str(e)}), 500

# Remove the __main__ block as it's not needed for SageMaker

# The rest of your code (functions, etc.) remains the same