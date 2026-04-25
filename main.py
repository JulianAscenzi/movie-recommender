import argparse  # This is a module that let me handle reading arguments from the terminal.
from src.recommender import MovieRecommender


def parse_args():
    parser = argparse.ArgumentParser(  # Creates a parser object
        description="Get movie recommendations based on a title."
    )
    parser.add_argument(  # Argument 1: title
        "title",
        type=str,
        help="The movie title to base recommendations on (e.g. 'toy story')",
    )
    parser.add_argument(  # Argument 2(optional): how many similar movies do you want to receive
        "--top",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()  # Gets the arguments

    print("Loading data...")
    recommender = MovieRecommender()
    recommender.load_and_prepare_data()

    print("Building model...")
    recommender.build_model()

    print(f"\nTop {args.top} recommendations for '{args.title}':\n")
    try:
        results = recommender.recommend(args.title, top_k=args.top)
        for i, (_, row) in enumerate(results.iterrows(), start=1):
            print(f"{i:>2}. {row['title']:<50} {row['genres']}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
