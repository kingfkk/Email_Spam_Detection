
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import ComplementNB
import joblib
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# STEP 1: PROJECT TITLE
# ==========================================

print("EMAIL SPAM DETECTION PROJECT")
print("----------------------------")


# ==========================================
# STEP 2: LOAD THE DATASET
# ==========================================

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# ==========================================
# STEP 3: DISPLAY FIRST FIVE MESSAGES
# ==========================================

print("\n========== FIRST FIVE MESSAGES ==========")
print(df.head())


# ==========================================
# STEP 4: DATASET INFORMATION
# ==========================================

print("\n========== DATASET INFORMATION ==========")
df.info()


# ==========================================
# STEP 5: COUNT SPAM AND HAM MESSAGES
# ==========================================

print("\n========== MESSAGE CATEGORIES ==========")
print(df["label"].value_counts())


# ==========================================
# STEP 6: DATA VISUALIZATION
# ==========================================

print("\n========== DATA VISUALIZATION ==========")

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

# Create spam vs ham bar chart
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="label")

plt.title("Spam vs Ham Message Distribution")
plt.xlabel("Message Category")
plt.ylabel("Number of Messages")

plt.savefig("graphs/message_distribution.png")

plt.show()

print("Graph saved in graphs folder!")


# ==========================================
# STEP 7: CHECK DATA QUALITY
# ==========================================

print("\n========== DATA QUALITY CHECK ==========")

# Check duplicate messages
print("\nDuplicate Messages:")
print(df.duplicated().sum())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)


# ==========================================
# STEP 8: DATA CLEANING
# ==========================================

print("\n========== DATA CLEANING ==========")

# Display original dataset size
print("Before cleaning:", df.shape)

# Remove duplicate rows
df_clean = df.drop_duplicates().copy()

# Display cleaned dataset size
print("After cleaning:", df_clean.shape)

# Check remaining duplicates
print("Remaining duplicates:", df_clean.duplicated().sum())

# Save cleaned dataset
df_clean.to_csv(
    "dataset/cleaned_messages.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


# ==========================================
# PROJECT PROGRESS
# ==========================================

print("\n================================")
print("STEPS 1 TO 8 COMPLETED!")
print("================================")

# ==========================================
# STEP 9: TEXT PREPROCESSING
# ==========================================

print("\n========== TEXT PREPROCESSING ==========")

# Function to clean message text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Apply preprocessing to cleaned messages
df_clean["clean_message"] = df_clean["message"].apply(clean_text)

# Display original and cleaned messages
print("\nOriginal Message:")
print(df_clean["message"].iloc[0])

print("\nPreprocessed Message:")
print(df_clean["clean_message"].iloc[0])

# Save preprocessed dataset
df_clean.to_csv(
    "dataset/preprocessed_messages.csv",
    index=False
)

print("\nPreprocessed dataset saved successfully!")

# ==========================================
# STEP 10: TF-IDF FEATURE EXTRACTION
# ==========================================

print("\n========== TF-IDF FEATURE EXTRACTION ==========")

# Separate messages (X) and labels (y)
X = df_clean["clean_message"]
y = df_clean["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Learn vocabulary from training messages only
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing messages
X_test_tfidf = tfidf.transform(X_test)

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

print("\nTF-IDF feature extraction completed successfully!")

# ==========================================
# STEP 11: MODEL TRAINING
# ==========================================

print("\n========== MODEL TRAINING ==========")

# Create Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model using training data
model.fit(X_train_tfidf, y_train)

print("Model training completed successfully!")

# Display model classes
print("Model classes:", model.classes_)

# ==========================================
# STEP 12: MODEL EVALUATION
# ==========================================

print("\n========== MODEL EVALUATION ==========")

# Predict messages from testing data
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Display precision, recall and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
print(cm)

print("\nModel evaluation completed successfully!")

# ==========================================
# STEP 13: TRY COMPLEMENT NAIVE BAYES
# ==========================================

print("\n========== COMPLEMENT NAIVE BAYES ==========")

# Create Complement Naive Bayes model
complement_model = ComplementNB()

# Train the model
complement_model.fit(X_train_tfidf, y_train)

# Make predictions
complement_pred = complement_model.predict(X_test_tfidf)

# Display evaluation
print("\nComplement Naive Bayes Classification Report:")
print(classification_report(y_test, complement_pred))

# Display confusion matrix
print("\nComplement Naive Bayes Confusion Matrix:")
complement_cm = confusion_matrix(
    y_test,
    complement_pred,
    labels=["ham", "spam"]
)

print(complement_cm)

print("\nComplement Naive Bayes training and evaluation completed!")

# ==========================================
# STEP 14: SAVE THE BEST MODEL
# ==========================================

print("\n========== SAVING BEST MODEL ==========")

# Create models folder if necessary
os.makedirs("models", exist_ok=True)

# Save Complement Naive Bayes model
joblib.dump(
    complement_model,
    "models/spam_model.pkl"
)

# Save TF-IDF vectorizer
joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("Spam detection model saved successfully!")
print("TF-IDF vectorizer saved successfully!")
print("Best model: Complement Naive Bayes")
