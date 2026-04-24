from src.data_loader import load_all


def test_load_all():
    movies, ratings = load_all()

    assert movies is not None
    assert ratings is not None
    assert len(movies) > 0
    assert len(ratings) > 0

    print(movies.head())
    print(ratings.head())
