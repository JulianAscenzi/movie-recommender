import os
import pandas as pd

DATA_PATH = "data"


def load_movies():
    file_path = os.path.join(
        DATA_PATH, "movies.csv"
    )  # Load movies.csv files and return it like DataFrame.

    if not os.path.exists(file_path):
        raise FileNotFoundError("File movies.csv not found")

    movies = pd.read_csv(file_path)

    print(f"movies.csv loaded correctly. rows: {len(movies)}")
    return movies


def load_ratings():

    file_path = os.path.join(
        DATA_PATH, "ratings.csv"
    )  # Load ratings.csv and return DataFrame.

    if not os.path.exists(file_path):
        raise FileNotFoundError("File ratings.csv not found")

    ratings = pd.read_csv(file_path)

    print(f"ratings.csv loaded correctly. rows: {len(ratings)}")
    return ratings


def load_all():  # Loads all the necesary tables for the sistem
    movies = load_movies()
    ratings = load_ratings()

    return movies, ratings
