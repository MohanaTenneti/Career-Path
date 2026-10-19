import streamlit as st
import pandas as pd
import joblib
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CareerPath",
    page_icon="??",
    layout="wide"
)

df = pd.read_csv("data/processed/occupation_profiles.csv")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
occupation_vectors = joblib.load(
    "models/tfidf_occupation_vectors.pkl"
)

SKILL_ALIASES = {
    "python": [
        "computer programming",
        "programming"
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
        "apply statistical analysis techniques"
    ],
    "data visualization": [
        "data visualisation",
        "data visualization"
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

    scores = cosine_similarity(
        user_vector,
        occupation_vectors
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    normalized_skills = normalize_skills(
        user_skills
    )

    results = []

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

        results.append({
            "career": occupation["occupation"],
            "similarity": float(scores[idx]),
            "matched": sorted(matched),
            "missing": sorted(missing)[:10]
        })

    return results


st.title("CareerPath")
st.subheader(
    "Personalized Career & Skill Recommendation"
)

st.write(
    "Enter your skills and interests to discover "
    "career paths that match your profile."
)

user_input = st.text_area(
    "Your skills",
    placeholder=(
        "Example: Python, machine learning, "
        "SQL, statistics, data analysis"
    ),
    height=100
)

top_k = st.slider(
    "Number of careers",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("Find Career Paths", type="primary"):

    user_skills = [
        skill.strip()
        for skill in user_input.split(",")
        if skill.strip()
    ]

    if not user_skills:

        st.warning(
            "Please enter at least one skill."
        )

    else:

        results = recommend_careers(
            user_skills,
            top_k
        )

        st.success(
            f"Found {len(results)} career paths."
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            st.markdown(
                f"### {i}. "
                f"{result['career'].title()}"
            )

            st.metric(
                "TF-IDF similarity",
                f"{result['similarity']:.3f}"
            )

            st.markdown("**Matched skills**")

            if result["matched"]:

                st.write(
                    ", ".join(result["matched"])
                )

            else:

                st.write("No direct skill matches found.")

            st.markdown("**Potential skill gaps**")

            if result["missing"]:

                st.write(
                    ", ".join(result["missing"])
                )

            else:

                st.write(
                    "No major skill gaps identified."
                )

            st.divider()

st.caption(
    "Recommendations are based on ESCO occupational "
    "skills and are intended as career guidance."
)
