import argparse

from lib.hybrid_search import normalize_scores, HybridSearch
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="*", type=float, help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="A search with weights"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Weight"
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            movies = load_movies()
            searcher = HybridSearch(movies)
            results = searcher.weighted_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
                print(f"  Hybrid Score: {res['hybrid_score']:.3f}")
                print(f"  BM25: {res['bm25_score']:.3f}, Semantic: {res['semantic_score']:.3f}")
                print(f"  {res['document'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
