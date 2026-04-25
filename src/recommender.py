import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.data_loader import load_all
from src.preprocessing import preprocess_movies


class MovieRecommender:
    """
    Memory-efficient content-based recommender using TF-IDF + linear_kernel.
    Computes similarity only for one movie at a time.
    """

    def __init__(self):
        self.movies = None
        self.tfidf = None
        self.tfidf_matrix = None

    def load_and_prepare_data(self):
        movies, _ = load_all()
        self.movies = preprocess_movies(movies)

    def build_model(self):
        # GUARD: check that data was loaded before trying to use it.
        # self.movies is set to None in __init__, and only gets a real
        # value after load_and_prepare_data() runs successfully.
        if self.movies is None:
            raise RuntimeError(
                "No data loaded. Call load_and_prepare_data() before build_model()."
            )
        # If we get here, self.movies exists and we can safely use it.
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["features"])

    def get_movie_index(self, title):
        clean_input = title.lower().strip()

        # 1. Try exact match first
        exact_matches = self.movies[self.movies["clean_title"] == clean_input]
        if len(exact_matches) == 1:
            return exact_matches.index[0]

        # 2. Fall back to substring match
        partial_matches = self.movies[
            self.movies["clean_title"].str.contains(clean_input, regex=False)
        ]

        if partial_matches.empty:
            raise ValueError(f"No movies found matching '{title}'.")

        if len(partial_matches) == 1:
            return partial_matches.index[0]

        # 3. Multiple matches — show options instead of picking silently
        options = partial_matches["title"].tolist()
        options_str = "\n  ".join(options)
        raise ValueError(
            f"Multiple movies found matching '{title}'. Please be more specific:\n  {options_str}"
        )

    def recommend(self, title, top_k=10):
        # GUARD: check that build_model() has been called.
        # self.tfidf_matrix is None until build_model() runs.
        if self.tfidf_matrix is None:
            raise RuntimeError(
                "Model not built. Call build_model() before recommend()."
            )

        # Safe to proceed — the matrix exists.
        idx = self.get_movie_index(title)

        # Get similarity scores only for the movie we care about
        sim_scores = linear_kernel(
            self.tfidf_matrix[idx : idx + 1], self.tfidf_matrix
        ).flatten()

        # Get top-k similar movies
        top_indices = sim_scores.argsort()[::-1][1 : top_k + 1]

        return self.movies.iloc[top_indices][["title", "genres"]]
