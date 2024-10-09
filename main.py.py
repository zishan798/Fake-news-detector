import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def verify_news(input_news, model, tfidf_vectorizer):
    # Preprocess the input news
    input_tfidf = tfidf_vectorizer.transform([input_news])

    # Predict the label for the input news
    label = model.predict(input_tfidf)

    if label[0] == 0:
        return "The news is classified as FAKE."
    else:
        return "The news is classified as TRUE."

# Load the Fake dataset
fake_df = pd.read_csv('Fake.csv')  # Replace 'fake_news_dataset.csv' with the actual name of your fake news dataset file

# Load the True dataset
true_df = pd.read_csv('True.csv')  # Replace 'true_news_dataset.csv' with the actual name of your true news dataset file

# Add a 'label' column to each dataset to indicate fake (0) or true (1)
fake_df['label'] = 0
true_df['label'] = 1

# Combine the datasets
combined_df = pd.concat([fake_df, true_df])

# Split the data into features (X) and labels (y)
X = combined_df['text']
y = combined_df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TfidfVectorizer and transform the text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression()

# model fit to training data
model.fit(tfidf_train, y_train)

# Evaluate the Model
y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# User input for news verification
user_input_news = input("Enter the news for verification: ")
verification_result = verify_news(user_input_news, model, tfidf_vectorizer)
print(verification_result)