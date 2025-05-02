import os
from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load models (use relative paths)
def load_models():
    """Load all ML models with error handling"""
    try:
        nb_model = joblib.load('models/nb_model.pkl')
        bilstm_model = load_model('models/bilstm_model.h5')
        tfidf = joblib.load('models/tfidf.pkl')
        with open('models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return nb_model, bilstm_model, tfidf, tokenizer
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

nb_model, bilstm_model, tfidf, tokenizer = load_models()

# Text preprocessing
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    """Robust text cleaning with error handling"""
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error cleaning text: {str(e)}")
        return ""

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = clean_text(text)
        
        try:
            # Naive Bayes prediction
            nb_features = tfidf.transform([cleaned_text])
            nb_prob = nb_model.predict_proba(nb_features)[0][1]
            
            # BiLSTM prediction
            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded_seq = pad_sequences(seq, maxlen=200, padding='post')
            lstm_prob = bilstm_model.predict(padded_seq, verbose=0)[0][0]
            
            # Hybrid prediction (weighted average)
            hybrid_prob = (0.4 * nb_prob) + (0.6 * lstm_prob)
            prediction = "Fake" if hybrid_prob > 0.5 else "Real"
            confidence = max(hybrid_prob, 1-hybrid_prob)
            
            return render_template('index.html', 
                               prediction_text=f'Result: {prediction}',
                               confidence_text=f'Confidence: {confidence:.2%}',
                               user_input=text)
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return render_template('index.html', 
                               prediction_text='Error processing request',
                               user_input=text)

# Heroku deployment settings
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
