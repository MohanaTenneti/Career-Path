import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

PROCESSED.mkdir(parents=True, exist_ok=True)

occupations = pd.read_csv(RAW / "occupations_en.csv")
relations = pd.read_csv(RAW / "occupationSkillRelations_en.csv")

occupation_info = occupations[
    ["conceptUri", "preferredLabel", "definition", "description"]
].copy()

occupation_info = occupation_info.rename(
    columns={
        "conceptUri": "occupationUri",
        "preferredLabel": "occupation",
    }
)

skills_by_occupation = (
    relations.groupby("occupationUri")["skillLabel"]
    .apply(lambda x: list(dict.fromkeys(x.dropna())))
    .reset_index()
)

dataset = occupation_info.merge(
    skills_by_occupation,
    on="occupationUri",
    how="inner"
)

dataset["skill_text"] = dataset["skillLabel"].apply(
    lambda skills: ", ".join(skills)
)

dataset["profile_text"] = (
    dataset["occupation"]
    + ". "
    + dataset["definition"].fillna("")
    + " Skills: "
    + dataset["skill_text"]
)

dataset = dataset[
    ["occupationUri", "occupation", "definition", "skillLabel", "skill_text", "profile_text"]
]

dataset.to_pickle(PROCESSED / "occupation_profiles.pkl")
dataset.to_csv(PROCESSED / "occupation_profiles.csv", index=False)

print("Created occupation profiles:", dataset.shape)
print("\nExample:")
print(dataset[["occupation", "skill_text"]].head(3).to_string(index=False))
