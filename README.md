# Movie Recommender

A content-based movie recommendation system built to practice Python, machine learning pipelines, and software engineering fundamentals.

## What it does

Given a movie title, it returns a list of similar movies based on title and genre using TF-IDF vectorization and cosine similarity.

```
$ recommend "toy story" --top 5

Loading data...
Building model...

Top 5 recommendations for 'toy story':

 1. Toy Story 2 (1999)                              Adventure|Animation|Children|Comedy|Fantasy
 2. A Bug's Life (1998)                             Adventure|Animation|Children|Comedy
 3. Monsters, Inc. (2001)                           Adventure|Animation|Children|Comedy|Fantasy
 4. Finding Nemo (2003)                             Adventure|Animation|Children|Comedy
 5. The Incredibles (2004)                          Action|Adventure|Animation|Children|Comedy
```

## What I learned building this

- Loading and preprocessing data with pandas
- Building a content-based recommendation system with scikit-learn
- Writing and organizing tests with pytest
- Structuring a Python project with `pyproject.toml`
- Handling edge cases and writing meaningful error messages
- Building a CLI with argparse

## Project structure

```
movie-recommender/
│
├── data/                   # CSV files (not tracked in git)
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── data_loader.py      # Loads movies.csv and ratings.csv
│   ├── preprocessing.py    # Cleans titles, processes genres, builds features
│   └── recommender.py      # TF-IDF model and recommendation logic
├── tests/
│   ├── test_load.py
│   ├── test_preprocessing.py
│   └── test_recommender.py
│
├── main.py                 # CLI entry point
├── pyproject.toml          # Project config and dependencies
└── README.md
```

## Installation

You need Python 3.10 or higher.

```bash
# Clone the repository
git clone https://github.com/JulianAscenzi/movie-recommender.git
cd movie-recommender

# Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux

# Install the project and its dependencies
pip install ".[dev]"
```

## Data

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/). Download it and place `movies.csv` and `ratings.csv` inside the `data/` folder.

## Usage

```bash
# Get 10 recommendations (default)
recommend "toy story"

# Get a custom number of recommendations
recommend "toy story" --top 5

# See all options
recommend --help
```

## Running the tests

```bash
pytest
```

## Dependencies

- pandas
- numpy
- scikit-learn
- scipy