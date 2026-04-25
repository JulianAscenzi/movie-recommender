import pytest
from src.recommender import MovieRecommender


def test_recommender_runs():  # This is the happy path test
    recommender = MovieRecommender()
    recommender.load_and_prepare_data()
    recommender.build_model()
    results = recommender.recommend("toy story", top_k=5)
    assert not results.empty
    assert len(results) == 5


def test_build_model_without_data_raises():  # Creates a fresh MovieRecommender and immediately calls build_model()
    recommender = MovieRecommender()
    with pytest.raises(RuntimeError, match="load_and_prepare_data"):
        recommender.build_model()


def test_recommend_without_model_raises():  # Loads data but skips build_model()
    recommender = MovieRecommender()
    recommender.load_and_prepare_data()
    with pytest.raises(RuntimeError, match="build_model"):
        recommender.recommend("toy story")


def test_recommend_unknown_movie_raises():  # Passes a nonsense string
    recommender = MovieRecommender()
    recommender.load_and_prepare_data()
    recommender.build_model()
    with pytest.raises(ValueError, match="No movies found"):
        recommender.recommend("xkqzwmvbnp")


def test_recommend_ambiguous_title_raises():  # Passes "toy story" as a substring that matches multiple movies (Toy Story 1,2,3...)
    recommender = MovieRecommender()
    recommender.load_and_prepare_data()
    recommender.build_model()
    with pytest.raises(ValueError, match="Please be more specific"):
        recommender.recommend("star wars")


def test_recommend_exact_title_returns_results():  # Passes the full exact title
    recommender = MovieRecommender()
    recommender.load_and_prepare_data()
    recommender.build_model()
    results = recommender.recommend("toy story", top_k=5)
    assert not results.empty
    assert len(results) == 5


def test_recommend_returns_correct_columns():  # Tests that the DataFrame has the columns "title" and "genre"
    recommender = MovieRecommender()
    recommender.load_and_prepare_data()
    recommender.build_model()
    results = recommender.recommend("toy story", top_k=3)
    assert "title" in results.columns
    assert "genres" in results.columns
