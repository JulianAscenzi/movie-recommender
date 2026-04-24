import pandas as pd


def clean_title(title):
    """
    I want a clean movie title so they are easier to work with.

    Example:
    "Toy Story (1995)" -> "toy story"
    """

    title = title.split("(")[0]  # remove year

    title = title.lower()  # lowercase everything

    title = title.strip()  # remove extra spaces

    return title


def process_titles(movies):

    movies["clean_title"] = movies["title"].apply(
        clean_title
    )  #  Apply title cleaning to the entire dataframe.

    return movies


def process_genres(movies):
    """
    Convert genres from string to list. -> "Action|Comedy" -> ["action", "comedy"]
    """

    movies["genres"] = movies["genres"].fillna("")  # if the movie doesn't have a genre

    movies["genres_list"] = movies["genres"].apply(lambda x: x.lower().split("|"))

    return movies


def create_features(movies):
    """
    Return the features of the movie, this will be usefull when we use similarity calculations.
    """

    movies["features"] = movies.apply(
        lambda row: row["clean_title"]
        + " "
        + " ".join(
            row["genres_list"]
        ),  # this takes the list and put the items together with a space
        axis=1,  # this is for Pandas to look at the entire row, one by one
    )
    return movies


def preprocess_movies(movies):
    """
    Run all the preprocessing steps in order.
    """

    movies = process_titles(movies)
    movies = process_genres(movies)
    movies = create_features(movies)

    return movies
