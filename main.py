import argparse
from src.hybrid import HybridRecommender


def parse_args():
    parser = argparse.ArgumentParser(
        description="Get movie recommendations based on a title."
    )
    parser.add_argument(
        "title",
        type=str,
        help="The movie title to base recommendations on (e.g. 'toy story')",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10)",
    )
    parser.add_argument(
        "--content-weight",
        type=float,
        default=0.5,
        help="Weight for content-based filtering, between 0 and 1 (default: 0.5)",
    )
    parser.add_argument(
        "--collab-weight",
        type=float,
        default=0.5,
        help="Weight for collaborative filtering, between 0 and 1 (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        recommender = HybridRecommender(
            content_weight=args.content_weight,
            collab_weight=args.collab_weight,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("Loading data...")
    recommender.load_and_prepare_data()

    print("Building model...")
    recommender.build_model()

    print(f"\nTop {args.top} recommendations for '{args.title}':\n")

    try:
        results = recommender.recommend(args.title, top_k=args.top)
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            genres_display = row["genres"] if row["genres"] != 0 else ""
            print(
                f"{i:>2}. {row['title']:<50} {genres_display:<40} score: {row['hybrid_score']:.3f}"
            )
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
