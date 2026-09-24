import pandas as pd
import random

# Load occupation-skill data
occupation_skills = pd.read_csv("datasets/occupation_skills.csv")

random.seed(42)

# Global pool of all available skills
all_skills = sorted(
    occupation_skills["Element Name"].dropna().unique().tolist()
)

candidate_records = []

for title, group in occupation_skills.groupby("Title"):

    required_skills = sorted(
        group["Element Name"].dropna().unique().tolist()
    )

    if len(required_skills) < 2:
        continue

    for _ in range(30):

        # Select relevant skills
        relevant_ratio = random.uniform(0.20, 0.90)

        relevant_count = max(
            1,
            round(len(required_skills) * relevant_ratio)
        )

        relevant_count = min(
            relevant_count,
            len(required_skills)
        )

        candidate_relevant = random.sample(
            required_skills,
            relevant_count
        )

        # Select some unrelated skills
        unrelated_pool = list(
            set(all_skills) - set(required_skills)
        )

        if unrelated_pool:
            unrelated_count = random.randint(0, min(5, len(unrelated_pool)))

            candidate_unrelated = random.sample(
                unrelated_pool,
                unrelated_count
            )
        else:
            candidate_unrelated = []

        # Complete candidate skill list
        candidate_skills = sorted(
            set(candidate_relevant + candidate_unrelated)
        )

        required_set = set(required_skills)
        candidate_set = set(candidate_skills)

        # Matching skills
        matched_skills = sorted(
            candidate_set & required_set
        )

        # Missing required skills
        missing_skills = sorted(
            required_set - candidate_set
        )

        # Transparent skill-match percentage
        match_percentage = round(
            len(matched_skills) / len(required_set) * 100,
            2
        )

        # Suitability classification
        if match_percentage < 40:
            suitability = "Low"
        elif match_percentage < 70:
            suitability = "Medium"
        else:
            suitability = "High"

        candidate_records.append({
            "Job Title": title,
            "Candidate Skills": ", ".join(candidate_skills),
            "Required Skills": ", ".join(required_skills),
            "Matched Skills": ", ".join(matched_skills),
            "Missing Skills": ", ".join(missing_skills),
            "Match Percentage": match_percentage,
            "Suitability": suitability
        })


# Create DataFrame
candidate_df = pd.DataFrame(candidate_records)

# Remove exact duplicate records
candidate_df = candidate_df.drop_duplicates().reset_index(drop=True)

# Save final dataset
candidate_df.to_csv(
    "datasets/candidate_matching_data.csv",
    index=False
)

print("Dataset created successfully!")
print("Rows:", len(candidate_df))
print("Columns:", candidate_df.columns.tolist())

print("\nMissing values:")
print(candidate_df.isnull().sum())

print("\nDuplicate rows:", candidate_df.duplicated().sum())

print("\nSuitability distribution:")
print(candidate_df["Suitability"].value_counts())

print("\nMatch percentage statistics:")
print(candidate_df["Match Percentage"].describe())