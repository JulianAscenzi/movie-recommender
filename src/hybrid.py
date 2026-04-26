import pandas as pd
from src.recommender import MovieRecommender
from src.collaborative import CollaborativeRecommender
from src.data_loader import load_all
from src.preprocessing import preprocess_movies


class HybridRecommender:
    # Hybrid recommender that combines content-based and collaborative filtering.
    # It uses a weighted average of normalized similarity scores.

    def __init__(
        # Default weights is 0.5/0.5 (equal trust)
        self,
        content_weight=0.5,
        collab_weight=0.5,
    ):

        # Verifies they add up to 1.0
        # We use 1e-6 because floating point arithmetic isn't exact
        if abs(content_weight + collab_weight - 1.0) > 1e-6:
            raise ValueError("content_weight and collab_weight must add up to 1.0")

        self.content_weight = content_weight
        self.collab_weight = collab_weight

        self.content_recommender = MovieRecommender()
        self.collab_recommender = CollaborativeRecommender()
        self.is_ready = False

    def load_and_prepare_data(self):
        movies, ratings = load_all()
        movies = preprocess_movies(movies)
        # We're directly setting the atribute because that would reload the files a second time.
        self.content_recommender.movies = movies
        self.collab_recommender.prepare(movies, ratings)

    def build_model(self):
        if self.content_recommender.movies is None:
            raise RuntimeError(
                "No data loaded. Call load_and_prepare_data() before build_model()."
            )

        self.content_recommender.build_model()
        self.collab_recommender.build_model()
        self.is_ready = True

    def _normalize(self, series):
        min_val = series.min()
        max_val = series.max()
        # This handles the situation where all scores are identical
        if max_val == min_val:
            return series.apply(lambda x: 1.0 if x > 0 else 0.0)

        return (series - min_val) / (max_val - min_val)

    def recommend(self, title, top_k=10):
        if not self.is_ready:
            raise RuntimeError(
                "Model not built. Call build_model() before recommend()."
            )
        # We ask each recommender for 3x more results
        # because after the outer merge, movies that appear in both lists will rank higher.
        # If we only fetched top_k from each, we might miss good candidates that ranked just outside the top 10.
        fetch_k = top_k * 3

        content_results = self.content_recommender.recommend(title, top_k=fetch_k)
        content_results = content_results.copy()
        content_results["content_score"] = self._normalize(
            pd.Series(range(len(content_results), 0, -1), index=content_results.index)
        )
        # Handles movies with no ratings data
        try:
            collab_results = self.collab_recommender.recommend(title, top_k=fetch_k)
            collab_results = collab_results.rename(
                columns={"similarity_score": "collab_score"}
            )
            collab_results["collab_score"] = self._normalize(
                collab_results["collab_score"]
            )
            has_collab = True
        except ValueError:
            has_collab = False

        if not has_collab:
            content_results["hybrid_score"] = content_results["content_score"]
            return content_results[["title", "genres", "hybrid_score"]].head(top_k)

        merged = pd.merge(
            content_results[["title", "genres", "content_score"]],
            collab_results[["title", "collab_score"]],
            on="title",
            how="outer",
        ).fillna(0)

        merged["hybrid_score"] = (
            self.content_weight * merged["content_score"]
            + self.collab_weight * merged["collab_score"]
        )

        merged = merged.sort_values("hybrid_score", ascending=False)

        return merged[["title", "genres", "hybrid_score"]].head(top_k)
