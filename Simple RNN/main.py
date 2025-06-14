# Step 1: Import Libraries and Load the model
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from keras.models import load_model
from keras.utils import CustomObjectScope
import numpy as np
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = tf.keras.models.load_model('./simple_rnn_imdb.h5', compile=False, safe_mode=False)

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i -3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Prediction Function
def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title('IMDB Moview Review Sentiment Analysis')
st.write('Enter a moview review to classify it as positive or negative')

# User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    
    # Make Prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write(f'Please enter a movie review')