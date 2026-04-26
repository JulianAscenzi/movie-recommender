import pandas as pd

import src.hybrid as hybrid_module
from src.hybrid import HybridRecommender


def sample_movies():
    return pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Toy Story (1995)",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
            },
            {
                "movieId": 2,
                "title": "Toy Story 2 (1999)",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
            },
            {
                "movieId": 3,
                "title": "Jumanji (1995)",
                "genres": "Adventure|Children|Fantasy",
            },
            {
                "movieId": 4,
                "title": "Grumpier Old Men (1995)",
                "genres": "Comedy|Romance",
            },
        ]
    )


def sample_ratings():
    return pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 5.0},
            {"userId": 1, "movieId": 2, "rating": 4.5},
            {"userId": 1, "movieId": 3, "rating": 2.0},
            {"userId": 2, "movieId": 1, "rating": 4.5},
            {"userId": 2, "movieId": 2, "rating": 5.0},
            {"userId": 2, "movieId": 4, "rating": 1.0},
            {"userId": 3, "movieId": 1, "rating": 2.0},
            {"userId": 3, "movieId": 3, "rating": 4.5},
            {"userId": 3, "movieId": 4, "rating": 4.0},
        ]
    )


def test_hybrid_recommender_runs_on_small_data(monkeypatch):
    monkeypatch.setattr(
        hybrid_module,
        "load_all",
        lambda: (sample_movies().copy(), sample_ratings().copy()),
    )

    recommender = HybridRecommender()
    recommender.load_and_prepare_data()
    recommender.build_model()
    results = recommender.recommend("toy story", top_k=2)

    assert recommender.collab_ready is True
    assert len(results) == 2
    assert "hybrid_score" in results.columns
    assert results.iloc[0]["title"] == "Toy Story 2 (1999)"


def test_hybrid_falls_back_to_content_when_collab_build_fails(monkeypatch):
    monkeypatch.setattr(
        hybrid_module,
        "load_all",
        lambda: (sample_movies().copy(), sample_ratings().copy()),
    )

    recommender = HybridRecommender()
    recommender.load_and_prepare_data()

    def raise_memory_error():
        raise MemoryError

    monkeypatch.setattr(recommender.collab_recommender, "build_model", raise_memory_error)

    recommender.build_model()
    results = recommender.recommend("toy story", top_k=2)

    assert recommender.collab_ready is False
    assert len(results) == 2
    assert "hybrid_score" in results.columns
