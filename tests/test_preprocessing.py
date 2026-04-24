from src.data_loader import load_all
from src.preprocessing import preprocess_movies


def test_preprocessing_pipeline():
    # load datasets
    movies, ratings = load_all()

    # run preprocessing
    movies = preprocess_movies(movies)

    # basic checks
    assert "clean_title" in movies.columns
    assert "genres_list" in movies.columns
    assert "features" in movies.columns

    # check a known transformation
    sample_title = movies.loc[movies["movieId"] == 1, "clean_title"].values[0]
    assert sample_title == "toy story"

    print(movies[["title", "clean_title", "genres_list", "features"]].head())
