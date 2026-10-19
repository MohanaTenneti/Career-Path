import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load processed occupation profiles
df = pd.read_csv("data/processed/occupation_profiles.csv")

# Load ISCO group information from the original ESCO occupations file
occupations = pd.read_csv("data/raw/occupations_en.csv")

isco = occupations[["conceptUri", "iscoGroup"]].copy()
isco = isco.rename(columns={"conceptUri": "occupationUri"})

df = df.merge(isco, on="occupationUri", how="left")

print("Occupations loaded:", len(df))

# Load TF-IDF model
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_vectors = joblib.load("models/tfidf_occupation_vectors.pkl")

# Load semantic model and embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("models/occupation_embeddings.npy")

# Keep only occupations with a valid ISCO group
df["iscoGroup"] = df["iscoGroup"].astype(str)
df = df[df["iscoGroup"] != "nan"].reset_index(drop=True)

# Only groups containing at least 2 occupations
group_counts = df["iscoGroup"].value_counts()
valid_groups = group_counts[group_counts >= 2].index
df = df[df["iscoGroup"].isin(valid_groups)].reset_index(drop=True)

# Match model rows to the filtered occupations
all_df = pd.read_csv("data/processed/occupation_profiles.csv")

original_indices = all_df[
    all_df["occupationUri"].isin(df["occupationUri"])
].index

tfidf_vectors = tfidf_vectors[original_indices]
embeddings = embeddings[original_indices]

rng = np.random.default_rng(42)


def average_precision_at_k(relevant, k):
    relevant = relevant[:k]

    if relevant.sum() == 0:
        return 0.0

    hits = 0
    score = 0.0

    for i, value in enumerate(relevant, start=1):
        if value:
            hits += 1
            score += hits / i

    return score / min(relevant.sum(), k)


def ndcg_at_k(relevant, k):
    relevant = relevant[:k]

    if relevant.sum() == 0:
        return 0.0

    discounts = 1 / np.log2(np.arange(2, len(relevant) + 2))
    dcg = np.sum(relevant * discounts)

    ideal_count = min(int(relevant.sum()), k)
    ideal_discounts = 1 / np.log2(
        np.arange(2, ideal_count + 2)
    )
    idcg = np.sum(ideal_discounts)

    return dcg / idcg


def evaluate_model(model_name):
    precision5 = []
    recall5 = []
    ndcg5 = []
    map5 = []

    precision10 = []
    recall10 = []
    ndcg10 = []
    map10 = []

    for i in range(len(df)):

        skills = df.loc[i, "skillLabel"]

        if not isinstance(skills, str) or not skills.strip():
            continue

        skill_list = [
            x.strip()
            for x in skills.split(",")
            if x.strip()
        ]

        if len(skill_list) < 4:
            continue

        # Hide 50% of the skills
        n_hidden = max(
            1,
            int(len(skill_list) * 0.5)
        )

        hidden_indices = rng.choice(
            len(skill_list),
            size=n_hidden,
            replace=False
        )

        visible_skills = [
            skill
            for j, skill in enumerate(skill_list)
            if j not in hidden_indices
        ]

        if len(visible_skills) < 2:
            continue

        query_text = ", ".join(visible_skills)

        if model_name == "TF-IDF":

            query_vector = vectorizer.transform(
                [query_text]
            )

            scores = cosine_similarity(
                query_vector,
                tfidf_vectors
            )[0]

        else:

            query_embedding = embedding_model.encode(
                [query_text],
                normalize_embeddings=True,
                show_progress_bar=False
            )

            scores = cosine_similarity(
                query_embedding,
                embeddings
            )[0]

        ranking = np.argsort(scores)[::-1]

        # Relevant careers are occupations
        # belonging to the same ISCO occupational group
        target_group = df.loc[i, "iscoGroup"]

        relevant_set = set(
            df.index[
                df["iscoGroup"] == target_group
            ].tolist()
        )

        relevance = np.array([
            1 if idx in relevant_set else 0
            for idx in ranking
        ])

        total_relevant = relevance.sum()

        if total_relevant == 0:
            continue

        precision5.append(
            relevance[:5].sum() / 5
        )

        recall5.append(
            relevance[:5].sum() / total_relevant
        )

        ndcg5.append(
            ndcg_at_k(relevance, 5)
        )

        map5.append(
            average_precision_at_k(relevance, 5)
        )

        precision10.append(
            relevance[:10].sum() / 10
        )

        recall10.append(
            relevance[:10].sum() / total_relevant
        )

        ndcg10.append(
            ndcg_at_k(relevance, 10)
        )

        map10.append(
            average_precision_at_k(relevance, 10)
        )

    return {
        "Model": model_name,
        "Queries": len(precision5),
        "Precision@5": np.mean(precision5),
        "Recall@5": np.mean(recall5),
        "NDCG@5": np.mean(ndcg5),
        "MAP@5": np.mean(map5),
        "Precision@10": np.mean(precision10),
        "Recall@10": np.mean(recall10),
        "NDCG@10": np.mean(ndcg10),
        "MAP@10": np.mean(map10)
    }


results = []

print("\nRunning TF-IDF evaluation...")
results.append(
    evaluate_model("TF-IDF")
)

print("Running semantic embedding evaluation...")
results.append(
    evaluate_model("Embeddings")
)

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MULTI-RELEVANCE EVALUATION")
print("=" * 70)
print("50% of occupation skills hidden")
print("Relevant careers = same ESCO/ISCO occupational group\n")

print(
    results_df.round(4).to_string(index=False)
)

results_df.to_csv(
    "data/processed/multi_relevance_results.csv",
    index=False
)

print(
    "\nSaved to "
    "data/processed/multi_relevance_results.csv"
)
