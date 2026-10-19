import pandas as pd
import numpy as np
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

rng = np.random.default_rng(42)

results = []

for idx, row in df.iterrows():

    skills = [
        skill.strip()
        for skill in str(row["skill_text"]).split(",")
        if skill.strip()
    ]

    if len(skills) < 5:
        continue

    n_hide = max(1, int(len(skills) * 0.3))

    hidden_indices = rng.choice(
        len(skills),
        size=n_hide,
        replace=False
    )

    visible_skills = [
        skill
        for i, skill in enumerate(skills)
        if i not in hidden_indices
    ]

    user_profile = " ".join(visible_skills)

    user_vector = vectorizer.transform([user_profile])

    scores = cosine_similarity(
        user_vector,
        occupation_vectors
    ).flatten()

    ranked_indices = np.argsort(scores)[::-1]

    target_uri = row["occupationUri"]

    ranked_uris = df.iloc[ranked_indices]["occupationUri"].values

    rank_positions = np.where(ranked_uris == target_uri)[0]

    if len(rank_positions) == 0:
        rank = len(ranked_uris) + 1
    else:
        rank = rank_positions[0] + 1

    results.append(rank)

results = np.array(results)

def recall_at_k(ranks, k):
    return np.mean(ranks <= k)

def ndcg_at_k(ranks, k):
    scores = np.where(
        ranks <= k,
        1 / np.log2(ranks + 1),
        0
    )
    return np.mean(scores)

print("\nEvaluation queries:", len(results))

print("\nTF-IDF Results")
print("-------------------------")
print(f"Recall@5:   {recall_at_k(results, 5):.4f}")
print(f"Recall@10:  {recall_at_k(results, 10):.4f}")
print(f"NDCG@5:     {ndcg_at_k(results, 5):.4f}")
print(f"NDCG@10:    {ndcg_at_k(results, 10):.4f}")

print("\nMedian rank:", int(np.median(results)))
print("Mean rank:", round(float(np.mean(results)), 2))
