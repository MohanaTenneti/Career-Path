import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

DATA = Path("data/processed/occupation_profiles.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating occupation embeddings...")

embeddings = model.encode(
    df["profile_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

np.save(
    MODEL_DIR / "occupation_embeddings.npy",
    embeddings
)

print("\nEmbeddings created.")
print("Number of occupations:", len(embeddings))
print("Embedding dimensions:", embeddings.shape[1])
print("Saved to:", MODEL_DIR / "occupation_embeddings.npy")
