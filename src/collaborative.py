from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeRecommender:
    """Item-based collaborative filtering built on a sparse ratings matrix."""

    def __init__(self):
        self.movies = None
        self.ratings = None
        self.item_matrix = None
        self.movie_ids = None
        self.movie_id_to_index = {}

    def prepare(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings

    def build_model(self):
        if self.ratings is None or self.movies is None:
            raise RuntimeError("No data loaded. Call prepare() before build_model().")

        ratings = self.ratings.groupby(["userId", "movieId"], as_index=False)[
            "rating"
        ].mean()

        user_codes = ratings["userId"].astype("category").cat.codes
        movie_categories = ratings["movieId"].astype("category")
        movie_codes = movie_categories.cat.codes

        self.movie_ids = movie_categories.cat.categories.to_list()
        self.movie_id_to_index = {
            movie_id: index for index, movie_id in enumerate(self.movie_ids)
        }
        self.item_matrix = csr_matrix(
            (
                ratings["rating"].to_numpy(),
                (movie_codes.to_numpy(), user_codes.to_numpy()),
            )
        )

    def get_movie_index(self, title):
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
            raise ValueError(f"No movies found matching '{title}'.")

        if len(partial_matches) == 1:
            return partial_matches.iloc[0]["movieId"]

        options = partial_matches["title"].tolist()
        options_str = "\n  ".join(options)
        raise ValueError(
            f"Multiple movies found matching '{title}'. Please be more specific:\n  {options_str}"
        )

    def recommend(self, title, top_k=10):
        if self.item_matrix is None:
            raise RuntimeError(
                "Model not built. Call build_model() before recommend()."
            )

        movie_id = self.get_movie_index(title)
        row_index = self.movie_id_to_index.get(movie_id)

        if row_index is None:
            raise ValueError(f"Movie '{title}' exists but has no ratings data.")

        sim_scores = cosine_similarity(
            self.item_matrix[row_index], self.item_matrix
        ).flatten()
        top_indices = sim_scores.argsort()[::-1][1 : top_k + 1]

        similar_movie_ids = [self.movie_ids[index] for index in top_indices]
        score_map = {
            movie_id: sim_scores[index]
            for movie_id, index in zip(similar_movie_ids, top_indices)
        }

        results = self.movies[self.movies["movieId"].isin(similar_movie_ids)][
            ["movieId", "title", "genres"]
        ].copy()
        results["similarity_score"] = results["movieId"].map(score_map)
        results = results.sort_values("similarity_score", ascending=False)

        return results[["title", "genres", "similarity_score"]]
