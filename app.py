
import re
import joblib
import streamlit as st

# Load the saved model and TF-IDF vectorizer
model = joblib.load("models/spam_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


# Clean the message using the same preprocessing as training
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Streamlit application
st.title("SMS Spam Detection")

st.write(
    "Enter a message below to check whether it is spam or normal."
)

message = st.text_area("Enter your message:")

if st.button("Check Message"):
    if message.strip():
        cleaned_message = clean_text(message)

        message_features = tfidf.transform([cleaned_message])

        prediction = model.predict(message_features)[0]

        if prediction == "spam":
            st.error("Warning! This message is SPAM.")
        else:
            st.success("This message is HAM (normal).")

    else:
        st.warning("Please enter a message.")
