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

user_profile = """
I am interested in machine learning, data analysis,
Python, statistics, SQL, predictive modelling,
data visualization and building intelligent systems.
"""

user_embedding = model.encode(
    [user_profile],
    normalize_embeddings=True
)

scores = cosine_similarity(
    user_embedding,
    occupation_embeddings
).flatten()

df["score"] = scores

recommendations = df.sort_values(
    "score",
    ascending=False
).head(10)

print("\nTop 10 semantic recommendations:\n")

for i, row in enumerate(recommendations.itertuples(), 1):
    print(f"{i}. {row.occupation} - {row.score:.4f}")
