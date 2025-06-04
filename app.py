import streamlit as st
import pickle

# Load the saved model
with open("college_sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("College Sentiment Predictor")
st.write("Enter the college name to predict the sentiment of reviews.")

# Input box for college name
college_input = st.text_input("College Name")

if st.button("Predict Sentiment"):
    if college_input.strip() == "":
        st.warning("Please enter a college name.")
    else:
        # Predict sentiment
        prediction = model.predict([college_input])[0]
        st.success(f"Predicted Sentiment: {prediction}")
