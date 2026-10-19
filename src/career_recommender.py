import pandas as pd
import joblib
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/processed/occupation_profiles.csv")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
occupation_vectors = joblib.load(
    "models/tfidf_occupation_vectors.pkl"
)

SKILL_ALIASES = {
    "python": [
        "computer programming",
        "programming",
        "software programming"
    ],
    "sql": [
        "database management systems",
        "database",
        "data storage"
    ],
    "data analysis": [
        "data analytics",
        "analyse big data",
        "data mining",
        "apply statistical analysis techniques"
    ],
    "machine learning": [
        "machine learning",
        "artificial intelligence",
        "predictive modelling"
    ],
    "statistics": [
        "statistics",
        "apply statistical analysis techniques",
        "statistical analysis"
    ],
    "data visualization": [
        "data visualisation",
        "data visualization",
        "create data visualisations"
    ]
}


def parse_skills(value):

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return [
                str(skill).strip()
                for skill in parsed
                if str(skill).strip()
            ]
    except (ValueError, SyntaxError):
        pass

    return []


def normalize_skills(user_skills):

    normalized = set()

    for skill in user_skills:

        skill = skill.lower().strip()
        normalized.add(skill)

        for alias in SKILL_ALIASES.get(skill, []):
            normalized.add(alias.lower())

    return normalized


def recommend_careers(user_skills, top_k=5):

    user_text = ", ".join(user_skills)

    user_vector = vectorizer.transform(
        [user_text]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        occupation_vectors
    )[0]

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]

    normalized_skills = normalize_skills(
        user_skills
    )

    recommendations = []

    for idx in top_indices:

        occupation = df.iloc[idx]

        occupation_skills = parse_skills(
            occupation["skillLabel"]
        )

        occupation_skill_set = {
            skill.lower()
            for skill in occupation_skills
        }

        matched = normalized_skills.intersection(
            occupation_skill_set
        )

        missing = occupation_skill_set.difference(
            normalized_skills
        )

        # Interpretable skill coverage score
        if len(occupation_skill_set) > 0:
            skill_match = (
                len(matched)
                / len(occupation_skill_set)
            ) * 100
        else:
            skill_match = 0

        recommendations.append({
            "career": occupation["occupation"],
            "similarity": round(
                float(similarity_scores[idx]), 4
            ),
            "skill_match": round(
                skill_match, 1
            ),
            "matched_skills": sorted(matched),
            "missing_skills": sorted(missing)[:10]
        })

    return recommendations


user_skills = [
    "Python",
    "machine learning",
    "data analysis",
    "statistics",
    "SQL",
    "data visualization"
]

results = recommend_careers(
    user_skills,
    top_k=5
)

print("\nPersonalized Career Recommendations")
print("=" * 60)

for i, result in enumerate(results, start=1):

    print(f"\n{i}. {result['career']}")
    print(
        f"TF-IDF similarity: {result['similarity']}"
    )
    print(
        f"Skill match: {result['skill_match']}%"
    )

    print(
        "Matched skills:",
        ", ".join(result["matched_skills"])
        if result["matched_skills"]
        else "None"
    )

    print(
        "Skill gaps:",
        ", ".join(result["missing_skills"])
        if result["missing_skills"]
        else "None"
    )
