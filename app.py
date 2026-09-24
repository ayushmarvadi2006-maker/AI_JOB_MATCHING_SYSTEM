from flask import Flask, request, jsonify, render_template
import joblib
import re

model = joblib.load("model_compressed.joblib")

app = Flask(__name__)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_skill_match(candidate_skills, required_skills):

    candidate_set = {
        skill.strip().lower()
        for skill in candidate_skills
        if skill.strip()
    }

    required_set = {
        skill.strip().lower()
        for skill in required_skills
        if skill.strip()
    }

    if not required_set:
        return 0.0, [], []

    matched_skills = sorted(candidate_set & required_set)
    missing_skills = sorted(required_set - candidate_set)

    percentage = round(
        len(matched_skills) / len(required_set) * 100,
        2
    )

    return percentage, matched_skills, missing_skills


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    candidate_skills = data.get("candidate_skills", [])
    required_skills = data.get("required_skills", [])

    if not candidate_skills or not required_skills:
        return jsonify({
            "error": "candidate_skills and required_skills are required"
        }), 400

    candidate_text = " ".join(candidate_skills)
    required_text = " ".join(required_skills)

    combined_text = candidate_text + " " + required_text
    cleaned_text = clean_text(combined_text)

    # Transparent skill matching
    match_percentage, matched_skills, missing_skills = calculate_skill_match(
        candidate_skills,
        required_skills
    )

    # Suitability based on match percentage
    if match_percentage < 40:
        prediction = "Low"
    elif match_percentage < 70:
        prediction = "Medium"
    else:
        prediction = "High"

    return jsonify({
        "suitability": prediction,
        "match_percentage": match_percentage,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    })


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )