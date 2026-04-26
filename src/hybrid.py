import pandas as pd
from src.recommender import MovieRecommender
from src.collaborative import CollaborativeRecommender
from src.data_loader import load_all
from src.preprocessing import preprocess_movies


class HybridRecommender:
    """Blend content-based and collaborative recommendations."""

    def __init__(
        self,
        content_weight=0.5,
        collab_weight=0.5,
    ):
        if abs(content_weight + collab_weight - 1.0) > 1e-6:
            raise ValueError("content_weight and collab_weight must add up to 1.0")

        self.content_weight = content_weight
        self.collab_weight = collab_weight

        self.content_recommender = MovieRecommender()
        self.collab_recommender = CollaborativeRecommender()
        self.collab_ready = False
        self.is_ready = False

    def load_and_prepare_data(self):
        movies, ratings = load_all()
        movies = preprocess_movies(movies)
        self.content_recommender.movies = movies
        self.collab_recommender.prepare(movies, ratings)
        self.collab_ready = False

    def build_model(self):
        if self.content_recommender.movies is None:
            raise RuntimeError(
                "No data loaded. Call load_and_prepare_data() before build_model()."
            )

        self.content_recommender.build_model()
        try:
            self.collab_recommender.build_model()
        except MemoryError:
            self.collab_ready = False
        else:
            self.collab_ready = True
        self.is_ready = True

    def _normalize(self, series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return series.apply(lambda x: 1.0 if x > 0 else 0.0)

        return (series - min_val) / (max_val - min_val)

    def recommend(self, title, top_k=10):
        if not self.is_ready:
            raise RuntimeError(
                "Model not built. Call build_model() before recommend()."
            )

        fetch_k = top_k * 3

        content_results = self.content_recommender.recommend(title, top_k=fetch_k)
        content_results = content_results.copy()
        content_results["content_score"] = self._normalize(
            pd.Series(range(len(content_results), 0, -1), index=content_results.index)
        )

        collab_results = None
        if self.collab_ready:
            try:
                collab_results = self.collab_recommender.recommend(title, top_k=fetch_k)
            except ValueError:
                collab_results = None

        if collab_results is None:
            content_results["hybrid_score"] = content_results["content_score"]
            return content_results[["title", "genres", "hybrid_score"]].head(top_k)

        collab_results = collab_results.rename(columns={"similarity_score": "collab_score"})
        collab_results["collab_score"] = self._normalize(collab_results["collab_score"])

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
