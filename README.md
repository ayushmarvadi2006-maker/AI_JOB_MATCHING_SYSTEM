# AI-Based Job Description Analysis and Candidate Skill Matching System

## Project Overview

This project is an AI-based web application that analyzes job-required skills and candidate skills to determine skill compatibility.

The system provides:

- Candidate suitability classification
- Skill match percentage
- Matched skills
- Missing skills
- Web-based user interface
- Flask API for prediction

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS
- JavaScript

## Machine Learning

The project uses:

- TF-IDF Vectorization for text feature extraction
- Random Forest Classifier for suitability classification

The suitability classes are:

- Low
- Medium
- High

## Dataset

The project uses O*NET-based occupational and software-skill information to construct the job skill dataset.

A candidate matching dataset was constructed for experimental machine-learning development.

## Model Evaluation

The final text-only model was evaluated using a group-aware train/test split based on job occupation.

Results:

- Accuracy: 65.55%
- Weighted Precision: 68.03%
- Weighted Recall: 65.55%
- Weighted F1-score: 65.04%

Training occupations: 735

Testing occupations: 184

Common occupations between training and testing: 0

## Application Workflow

1. User enters job-required skills.
2. User enters candidate skills.
3. Input text is cleaned.
4. The trained machine-learning model predicts suitability.
5. The system calculates skill overlap.
6. Matched and missing skills are identified.
7. Match percentage is displayed.
8. Results are presented through the web interface.

## Running the Application

Activate the virtual environment and run:

```bash
python app.py