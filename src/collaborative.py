import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:  # Item-based collaborative filtering recommender.

    def __init__(self):
        self.movies = None
        self.ratings = None
        self.movie_matrix = None
        self.similarity_matrix = None

    def prepare(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings

    def build_model(self):
        if self.ratings is None or self.movies is None:
            raise RuntimeError("No data loaded. Call prepare() before build_model().")

        self.movie_matrix = self.ratings.pivot_table(  # Rows are userId, columns are movieId, values are rating
            index="userId", columns="movieId", values="rating"
        ).fillna(
            0
        )  # This replaces missing ratings with Zero

        self.similarity_matrix = cosine_similarity(
            self.movie_matrix.T
        )  # .T transposes the matrix so we're comparing columns (movies) against each other

    def get_movie_index(
        self, title
    ):  # Works the same as in MovieRecommender but returns movieId
        if self.movies is None:
            raise RuntimeError(
                "No data loaded. Call prepare() before get_movie_index()."
            )

        clean_input = title.lower().strip()

        exact_matches = self.movies[self.movies["clean_title"] == clean_input]
        if len(exact_matches) == 1:
            return exact_matches.iloc[0]["movieId"]

        partial_matches = self.movies[
            self.movies["clean_title"].str.contains(clean_input, regex=False)
        ]

        if partial_matches.empty:
            raise ValueError(
                f"No movies found matching '{title}'."
            )  # No movie to recommend

        if len(partial_matches) == 1:
            return partial_matches.iloc[0]["movieId"]

        options = partial_matches["title"].tolist()
        options_str = "\n  ".join(options)
        raise ValueError(
            f"Multiple movies found matching '{title}'. Please be more specific:\n  {options_str}"
        )

    def recommend(self, title, top_k=10):
        if self.similarity_matrix is None:
            raise RuntimeError(
                "Model not built. Call build_model() before recommend()."
            )

        movie_id = self.get_movie_index(title)

        if (
            movie_id not in self.movie_matrix.columns
        ):  # Checks if the movie exists the columns. A movie can have zero ratings
            raise ValueError(f"Movie '{title}' exists but has no ratings data.")

        col_index = self.movie_matrix.columns.get_loc(
            movie_id
        )  # Converts a movieId value into a positional index

        sim_scores = list(enumerate(self.similarity_matrix[col_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : top_k + 1]

        similar_movie_ids = [self.movie_matrix.columns[i] for i, _ in sim_scores]
        scores = [score for _, score in sim_scores]

        results = self.movies[self.movies["movieId"].isin(similar_movie_ids)][
            ["movieId", "title", "genres"]
        ].copy()

        # similarity_score is a new column. It's a number between 0 and 1
        # 1 means identical rating patterns and 0 means no overlap at all
        score_map = dict(zip(similar_movie_ids, scores))
        results["similarity_score"] = results["movieId"].map(score_map)
        results = results.sort_values("similarity_score", ascending=False)

        return results[["title", "genres", "similarity_score"]]
