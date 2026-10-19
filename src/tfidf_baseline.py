import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/processed/occupation_profiles.csv")

df = pd.read_csv(DATA)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

occupation_vectors = vectorizer.fit_transform(df["profile_text"])

user_profile = """
Python, machine learning, statistics, data analysis,
SQL, predictive modelling, data visualization
"""

user_vector = vectorizer.transform([user_profile])

scores = cosine_similarity(
    user_vector,
    occupation_vectors
).flatten()

df["score"] = scores

recommendations = df.sort_values(
    "score",
    ascending=False
).head(10)

print("\nTop 10 career recommendations:\n")

for i, row in enumerate(recommendations.itertuples(), 1):
    print(f"{i}. {row.occupation} - {row.score:.4f}")
