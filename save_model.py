import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# Load processed dataset
df = pd.read_csv("datasets/processed_candidate_data.csv")

X = df["Cleaned Text"]
y = df["Suitability"]


# Text-based ML pipeline
model = Pipeline(steps=[
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])


# Train model on complete dataset
model.fit(X, y)


# Save trained model
joblib.dump(model, "model.joblib")

print("Final text-only model trained and saved successfully.")