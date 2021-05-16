from src.python.services import *
from src.python.module_factory import ModuleFactory
import argparse
import os
import sys
import subprocess
from src.python.code_analyzer.code_analyzer import AnalysisLevel
from src.python.entities.entity import Language
from pathlib import Path


def dataset(args):
    DataCollectionService.create_dataset(args)


def fetch_source_code_if_needed(args):
    path = args.project_path
    slug = args.project_slug
    if path is None and slug is None:
        print(
            f"At least one of the --project-path or --project-slug should be provided."
        )
        sys.exit()
    if path is None:
        name = slug.split("/")[-1]
        args.project_path = args.output_path / name
        if args.project_path.exists():
            return
        clone_command = f"git clone https://github.com/{slug}.git {args.project_path}"
        return_code = subprocess.call(clone_command, shell=True)
        if return_code != 0:
            print("Failure in fetching source code for GitHub!")
            sys.exit()


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
        help="Specifies the directory to save resulting datasets.",
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

    add_common_arguments(dataset_parser)
    dataset_parser.set_defaults(func=dataset)
    dataset_parser.add_argument(
        "-n",
        "--build-window",
        help="Specifies the number of recent builds to consider for computing features.",
        type=int,
        required=True,
    )

    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    fetch_source_code_if_needed(args)
    args.unique_separator = "\t"
    args.func(args)


if __name__ == "__main__":
    main()
