import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

DATA = Path("data/processed/occupation_profiles.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

occupation_vectors = vectorizer.fit_transform(df["profile_text"])

joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

joblib.dump(occupation_vectors, MODEL_DIR / "tfidf_occupation_vectors.pkl")

df.to_pickle(MODEL_DIR / "occupation_profiles.pkl")

print("TF-IDF baseline saved.")
print("Occupations indexed:", len(df))
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Vector shape:", occupation_vectors.shape)
