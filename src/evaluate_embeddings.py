import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/processed/occupation_profiles.csv")
EMBEDDINGS = Path("models/occupation_embeddings.npy")

df = pd.read_csv(DATA)
occupation_embeddings = np.load(EMBEDDINGS)

model = SentenceTransformer("all-MiniLM-L6-v2")

rng = np.random.default_rng(42)

def evaluate(hidden_fraction):

    ranks = []

    for _, row in df.iterrows():

        skills = [
            skill.strip()
            for skill in str(row["skill_text"]).split(",")
            if skill.strip()
        ]

        if len(skills) < 5:
            continue

        n_hide = max(1, int(len(skills) * hidden_fraction))

        hidden = rng.choice(
            len(skills),
            size=n_hide,
            replace=False
        )

        visible = [
            skill
            for i, skill in enumerate(skills)
            if i not in hidden
        ]

        user_embedding = model.encode(
            [" ".join(visible)],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            user_embedding,
            occupation_embeddings
        ).flatten()

        ranked = np.argsort(scores)[::-1]

        target_uri = row["occupationUri"]

        ranked_uris = df.iloc[ranked]["occupationUri"].values

        position = np.where(ranked_uris == target_uri)[0]

        if len(position):
            ranks.append(position[0] + 1)

    ranks = np.array(ranks)

    recall5 = np.mean(ranks <= 5)
    recall10 = np.mean(ranks <= 10)

    ndcg5 = np.mean(
        np.where(
            ranks <= 5,
            1 / np.log2(ranks + 1),
            0
        )
    )

    ndcg10 = np.mean(
        np.where(
            ranks <= 10,
            1 / np.log2(ranks + 1),
            0
        )
    )

    return recall5, recall10, ndcg5, ndcg10, np.median(ranks)


for hidden_fraction in [0.30, 0.50, 0.70]:

    recall5, recall10, ndcg5, ndcg10, median_rank = evaluate(
        hidden_fraction
    )

    print(f"\nHidden skills: {int(hidden_fraction * 100)}%")
    print("--------------------------------")
    print(f"Recall@5:   {recall5:.4f}")
    print(f"Recall@10:  {recall10:.4f}")
    print(f"NDCG@5:     {ndcg5:.4f}")
    print(f"NDCG@10:    {ndcg10:.4f}")
    print(f"Median rank: {median_rank:.0f}")
