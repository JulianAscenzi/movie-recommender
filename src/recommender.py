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
        """
        Build the TF-IDF vectors for all movies.
        Does NOT compute full similarity matrix (too big).
        """
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["features"])

    def get_movie_index(self, title):
        title = title.lower()
        matches = self.movies[self.movies["clean_title"].str.contains(title)]
        if matches.empty:
            raise ValueError(f"Movie '{title}' not found.")
        return matches.index[0]

    def recommend(self, title, top_k=10):
        """
        Compute similarity ONLY for the target movie.
        Uses linear_kernel, which is efficient for sparse TF-IDF.
        """

        idx = self.get_movie_index(title)

        # Get similarity scores only for the movie we care about
        sim_scores = linear_kernel(
            self.tfidf_matrix[idx : idx + 1], self.tfidf_matrix
        ).flatten()

        # Get top-k similar movies
        top_indices = sim_scores.argsort()[::-1][1 : top_k + 1]

        return self.movies.iloc[top_indices][["title", "genres"]]
