import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("balanced_college_reviews.csv")
# Drop rows where 'college' or 'text_sentiment' is missing
df = df.dropna(subset=['college', 'text_sentiment'])

# Now continue as before


# Check required columns
print(df.columns)

# Train model using college name as feature and text_sentiment as label
X = df['college']  # Feature: college name
y = df['text_sentiment']  # Label: sentiment

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model for Streamlit later (optional)
import pickle
with open("college_sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
