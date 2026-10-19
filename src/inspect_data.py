import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

occupations = pd.read_csv(DATA_DIR / "occupations_en.csv")
skills = pd.read_csv(DATA_DIR / "skills_en.csv")
relations = pd.read_csv(DATA_DIR / "occupationSkillRelations_en.csv")

print("Occupations:", occupations.shape)
print("Skills:", skills.shape)
print("Relations:", relations.shape)

print("\nOccupation columns:")
print(occupations.columns.tolist())

print("\nSkill columns:")
print(skills.columns.tolist())

print("\nRelation columns:")
print(relations.columns.tolist())

print("\nFirst occupation:")
print(occupations.head(1).to_string())

print("\nFirst skill:")
print(skills.head(1).to_string())

print("\nFirst relation:")
print(relations.head(1).to_string())
