# CareerPath

CareerPath is a career recommendation project that recommends occupations based on a user's skills and interests.

I built this project using the ESCO dataset, which contains occupations and the skills associated with them. The main idea was to see how well a user's skill profile could be matched to different careers and also show the skills that may be useful for each recommended career.

## What it does

The user enters skills such as:

Python, machine learning, SQL, statistics, data analysis

CareerPath then:

- ranks relevant occupations
- shows a similarity score
- shows matched skills
- shows potential skill gaps

The project also has a Streamlit interface for testing the recommendations interactively.

## Dataset

I used the English ESCO dataset from the European Commission.

The version used in this project contains:

- 3,043 occupations
- 13,960 skills
- 126,051 occupation-skill relationships

The occupation and skill relationship files were combined to create a skill profile for each occupation.

## How I approached it

I started with a TF-IDF based approach.

Each occupation was represented using its occupation information and associated skills. The user's skills were converted into the same TF-IDF representation and cosine similarity was used to rank the occupations.

I also tried a pretrained Sentence Transformer model (all-MiniLM-L6-v2) to see whether semantic embeddings would give better recommendations.

Interestingly, the embedding model did not perform better than TF-IDF on my offline evaluation. Since the ESCO data has fairly explicit skill information, the simpler TF-IDF representation worked better for this particular setup.

## Evaluation

I used two different types of offline evaluation.

The first one hid some of an occupation's skills and checked whether the original occupation could still be retrieved.

I also created a second evaluation where occupations belonging to the same ESCO/ISCO occupational group were treated as related careers. This gives multiple relevant careers instead of treating only one occupation as the correct answer.

For the second evaluation, 50% of the skills were hidden.

| Model | Precision@5 | Recall@5 | NDCG@5 | MAP@5 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 0.1748 | 0.0878 | 0.4594 | 0.4038 |
| Embeddings | 0.1466 | 0.0755 | 0.4001 | 0.3505 |

These numbers are from an offline evaluation using ESCO data. They don't represent real-world hiring or career-success probabilities.

## Skill matching

One issue I noticed while building the project was that people and datasets don't always use the same words for a skill.

For example, a user might enter Python, while the ESCO data may describe related skills using terms such as computer programming.

I added a small skill normalization layer for some common technical skills so that these differences don't completely prevent a match.

This is still a simple version and could be improved with a larger skill vocabulary.

## Project structure

CareerPath/
|
├── app.py
├── requirements.txt
├── README.md
|
├── data/
│   └── processed/
|
├── models/
|
└── src/

The src folder contains the scripts used for preprocessing, building the TF-IDF model, creating embeddings and running the evaluations.

## How to run

Install the required packages:

pip install -r requirements.txt

Then run the Streamlit application:

streamlit run app.py

Enter your skills separated by commas.

For example:

Python, machine learning, SQL, statistics, data analysis

## Limitations

There are a few limitations with the current version.

- The recommendations depend on the skills available in ESCO.
- Skill names are not always the same as the terms people normally use.
- The skill normalization currently covers only a small set of common aliases.
- The evaluation is based on ESCO rather than real user preferences or job outcomes.
- The system does not use collaborative filtering because there is no user-career interaction data in the dataset.
- A career recommendation should not be treated as a prediction of whether someone will get a particular job.

## Possible improvements

Some things I would like to improve later are:

- better skill normalization
- incorporating user preferences such as interests and education
- testing additional ranking methods
- improving the recommendation explanations
- getting human feedback on the recommendations

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- PyTorch
- Joblib
- Streamlit

## Data source

ESCO - European Skills, Competences, Qualifications and Occupations, European Commission.
