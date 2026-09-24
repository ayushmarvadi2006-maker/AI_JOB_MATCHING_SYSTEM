import pandas as pd
import re

# Load the candidate matching dataset
df = pd.read_csv("datasets/candidate_matching_data.csv")

# Combine candidate skills and required skills
df["Combined Text"] = (
    df["Candidate Skills"].fillna("") + " " +
    df["Required Skills"].fillna("")
)

# Basic text preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["Cleaned Text"] = df["Combined Text"].apply(clean_text)

# Keep only the columns needed for ML
processed_df = df[
    [
        "Job Title",
        "Candidate Skills",
        "Required Skills",
        "Cleaned Text",
        "Match Percentage",
        "Suitability"
    ]
].copy()

# Save processed dataset
processed_df.to_csv(
    "datasets/processed_candidate_data.csv",
    index=False
)

print("Preprocessing completed successfully!")
print("Rows:", len(processed_df))
print("Columns:", processed_df.columns.tolist())

print("\nSuitability distribution:")
print(processed_df["Suitability"].value_counts())

print("\nSample cleaned text:")
print(processed_df[["Job Title", "Cleaned Text", "Suitability"]].head(3).to_string(index=False))