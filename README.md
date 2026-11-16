# SMS Spam Detection Using Machine Learning

## 1. Project Overview

This project uses Machine Learning and Natural Language Processing (NLP) to classify SMS messages into two categories:

- **Spam:** Unwanted or promotional messages.
- **Ham:** Normal, legitimate messages.

The project uses the SMS Spam Collection dataset and a Complement Naive Bayes classification model.

## 2. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## 3. Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

- Original messages: 5,572
- Duplicate messages removed: 403
- Cleaned messages: 5,169
- Classes: Spam and Ham

## 4. Project Workflow

1. Load the SMS dataset.
2. Explore and analyze the data.
3. Check missing values and duplicate records.
4. Remove duplicate messages.
5. Clean and preprocess text.
6. Split the dataset into training and testing sets.
7. Convert text into numerical features using TF-IDF.
8. Train Naive Bayes classification models.
9. Evaluate model performance.
10. Save the trained model and TF-IDF vectorizer.
11. Build a web application using Streamlit.

## 5. Machine Learning Model

Two models were evaluated:

- Multinomial Naive Bayes
- Complement Naive Bayes

Complement Naive Bayes was selected because it achieved better spam detection performance in the comparison.

### Model Performance

- Accuracy: Approximately 97.10%
- Spam Precision: 0.90
- Spam Recall: 0.86
- Spam F1-score: 0.88

## 6. Project Structure

```text
Email_Spam_Detection/
│
├── dataset/
│   ├── SMSSpamCollection
│   ├── cleaned_messages.csv
│   └── preprocessed_messages.csv
│
├── graphs/
│   └── message_distribution.png
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
├── venv/
├── main.py
├── app.py
└── README.md
```

## 7. How to Run the Project

### Step 1: Activate the virtual environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 2: Install required libraries

```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### Step 3: Run the Streamlit application

```powershell
python -m streamlit run app.py
```

### Step 4: Use the application

Enter an SMS message and click the **Check Message** button to see whether the message is classified as Spam or Ham.

## 8. Conclusion

This project demonstrates how Machine Learning and Natural Language Processing can be used to detect unwanted SMS messages. The trained model is integrated into a Streamlit web application for interactive message classification.

## 9. Future Improvements

- Test additional Machine Learning algorithms.
- Improve spam detection performance.
- Add more data visualizations.
- Evaluate the model using cross-validation.
- Deploy the application online.
