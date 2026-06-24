import argparse

from lib.hybrid_search import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")


    hybrid_normalize_scores_parser = subparsers.add_parser(
        "normalize", help="Normalizes the scores"
    )
    hybrid_normalize_scores_parser.add_argument(
        "scores", type=float, nargs="*", help="List of scores"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize_scores(args.scores)
            if len(results) != 0:
                for score in results:
                    print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()