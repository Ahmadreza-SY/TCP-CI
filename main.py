from src.python.services import *
import argparse
from src.python.code_analyzer.code_analyzer import AnalysisLevel
from src.python.entities.entity import Language
from pathlib import Path


def dataset(args):
    DataCollectionService.create_dataset(args)


def learn(args):
    DataCollectionService.run_all_tsp_accuracy_experiments(args)


def add_common_arguments(parser):
    parser.add_argument(
        "-p",
        "--project-path",
        help="Project's source code git repository path.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--project-slug",
        help="The project's GitHub slug, e.g., apache/commons.",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--test-path",
        help="Specifies the relative root directory of the test source code.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-l",
        "--level",
        help="Specifies the granularity of feature extraction.",
        type=AnalysisLevel,
        choices=list(AnalysisLevel),
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    parser.add_argument(
        "--language",
        help="Project's main language",
        type=Language,
        choices=list(Language),
        required=True,
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Create training dataset including all test case features for each CI cycle.",
    )
    learn_parser = subparsers.add_parser(
        "learn",
        help="Perform learning experiments on collected features using RankLib.",
    )

    add_common_arguments(dataset_parser)
    dataset_parser.set_defaults(func=dataset)
    dataset_parser.add_argument(
        "-n",
        "--build-window",
        help="Specifies the number of recent builds to consider for computing features.",
        type=int,
        required=True,
    )

    learn_parser.set_defaults(func=learn)
    learn_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )

    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.unique_separator = "\t"
    args.func(args)


if __name__ == "__main__":
    main()
