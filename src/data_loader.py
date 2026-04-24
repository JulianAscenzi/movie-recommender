import logging
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"

logger = logging.getLogger(__name__)


def load_movies():
    """Load movies.csv and return it as a DataFrame."""
    file_path = DATA_PATH / "movies.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    import pandas as pd

    movies = pd.read_csv(file_path)
    logger.info(f"movies.csv loaded correctly. rows: {len(movies)}")
    return movies


def load_ratings():
    """Load ratings.csv and return it as a DataFrame."""
    file_path = DATA_PATH / "ratings.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    import pandas as pd

    ratings = pd.read_csv(file_path)
    logger.info(f"ratings.csv loaded correctly. rows: {len(ratings)}")
    return ratings


def load_all():
    """Load all the necessary tables for the system."""
    movies = load_movies()
    ratings = load_ratings()
    return movies, ratings
