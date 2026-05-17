import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "phishing_detector.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def get_mock_data():
    data = {
        "text": [
            "Verify your banking details immediately to avoid account suspension.",
            "Hey, are we still meeting for lunch at 12:30 PM today?",
            "URGENT: Your Netflix subscription has expired. Click here to renew.",
            "Please review the attached project proposal for Q3.",
            "Dear customer, suspicious activity detected. Log in at secure-bank-login.com",
            "The weekly newsletter is ready. Check out the latest updates.",
            "Congratulations! You won a $1000 Amazon gift card. Claim now!",
            "Can you send me the documentation for the API integration?"
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

def train_model():
    df = get_mock_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42
    )
    
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train_vec, y_train)
    
    predictions = model.predict(X_test_vec)
    
    with open(MODEL_PATH, "wb") as m_file:
        pickle.dump(model, m_file)
    with open(VECTORIZER_PATH, "wb") as v_file:
        pickle.dump(vectorizer, v_file)

def predict_phishing(input_text):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_model()
        
    with open(MODEL_PATH, "rb") as m_file:
        model = pickle.load(m_file)
    with open(VECTORIZER_PATH, "rb") as v_file:
        vectorizer = pickle.load(v_file)
        
    vectorized_text = vectorizer.transform([input_text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0][prediction]
    
    result = "🚨 PHISHING DETECTED" if prediction == 1 else "🟢 SAFE"
    print(f"\nResult: {result} ({probability*100:.2f}% confidence)")
    print(f"Text: \"{input_text}\"")

if __name__ == "__main__":
    train_model()
    predict_phishing("Urgent: Reset your password by clicking this link immediately!")
    predict_phishing("Let's sync up tomorrow morning to discuss the budget.")
    
