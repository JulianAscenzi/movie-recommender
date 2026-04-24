from src.recommender import MovieRecommender


def test_recommender_runs():
    recommender = MovieRecommender()

    recommender.load_and_prepare_data()
    recommender.build_model()

    results = recommender.recommend("toy story", top_k=5)

    assert not results.empty
    assert len(results) == 5
