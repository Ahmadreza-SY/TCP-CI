import os

os.environ["OUTDATED_IGNORE"] = "1"

from src.python.services.data_collection_service import DataCollectionService
from src.python.services.experiments_service import ExperimentsService, Experiment
import argparse
from src.python.code_analyzer.code_analyzer import AnalysisLevel
from src.python.entities.entity import Language
from pathlib import Path
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def dataset(args):
    DataCollectionService.create_dataset(args)


def tr_torrent(args):
    DataCollectionService.process_tr_torrent(args)


def learn(args):
    if args.ranking_models == "best":
        ExperimentsService.run_best_ranker_experiments(args)
    elif args.ranking_models == "all":
        ExperimentsService.run_all_tcp_rankers(args)


def hypopt(args):
    ExperimentsService.hyp_param_opt(args)


def decay_test(args):
    ExperimentsService.run_decay_test_experiments(args)


def results(args):
    ExperimentsService.analyze_results(args)


def add_dataset_parser_arguments(parser):
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
        "-c",
        "--ci-data-path",
        help="Path to CI datasource root directory, including RTP-Torrent and Travis Torrent.",
        type=Path,
        required=True,
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
        choices=[AnalysisLevel.FILE],
        default=AnalysisLevel.FILE,
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
        choices=[Language.JAVA],
        default=Language.JAVA,
    )
    parser.add_argument(
        "-n",
        "--build-window",
        help="Specifies the number of recent builds to consider for computing features.",
        type=int,
        required=False,
        default=6,
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    tr_torrent_parser = subparsers.add_parser(
        "tr_torrent",
        help="Process travis torrent logs and build info.",
    )
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Create training dataset including all test case features for each CI cycle.",
    )
    learn_parser = subparsers.add_parser(
        "learn",
        help="Perform learning experiments on collected features using RankLib.",
    )
    hypopt_parser = subparsers.add_parser(
        "hypopt",
        help="Perform hyperparameter optimization for the best ML ranking model in RankLib.",
    )
    decay_test_parser = subparsers.add_parser(
        "decay_test",
        help="Perform ML ranking models decay test experiments on trained models.",
    )
    results_parser = subparsers.add_parser(
        "results",
        help="Analyze the results from experiments and generate tables.",
    )

    add_dataset_parser_arguments(dataset_parser)
    dataset_parser.set_defaults(func=dataset)

    tr_torrent_parser.set_defaults(func=tr_torrent)
    tr_torrent_parser.add_argument(
        "-r",
        "--repo",
        help="The login and name of the repo seperated by @ (e.g., presto@prestodb)",
        type=str,
        required=True,
    )
    tr_torrent_parser.add_argument(
        "-i",
        "--input-path",
        help="Specifies the directory to of travis torrent raw data.",
        type=Path,
        required=True,
    )
    tr_torrent_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save resulting data.",
        type=Path,
        default=".",
    )

    learn_parser.set_defaults(func=learn)
    learn_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    learn_parser.add_argument(
        "-t",
        "--test-count",
        help="Specifies the number of recent builds to test the trained models on.",
        type=int,
        default=50,
    )
    learn_parser.add_argument(
        "-r",
        "--ranking-models",
        help="Specifies the ranking model(s) to use for learning.",
        type=str,
        default="best",
        choices=["best", "all"],
    )
    learn_parser.add_argument(
        "-e",
        "--experiment",
        help="Specifies the experiment to run. Only works when the best ranking model is selected.",
        type=Experiment,
        default=Experiment.FULL,
        choices=[e for e in Experiment],
    )

    hypopt_parser.set_defaults(func=hypopt)
    hypopt_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory of all datasets.",
        type=Path,
        default=".",
    )
    hypopt_parser.add_argument(
        "-b",
        "--build",
        help="Specifies the build id for running the optimization.",
        type=int,
        required=True,
    )
    hypopt_parser.add_argument(
        "-i",
        "--comb-index",
        help="Specifies the index of the hyperparameter combination.",
        type=int,
        required=True,
    )

    decay_test_parser.set_defaults(func=decay_test)
    decay_test_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save and load resulting datasets.",
        type=Path,
        default=".",
    )
    decay_test_parser.add_argument(
        "-p",
        "--project-path",
        help="Project's source code git repository path.",
        type=Path,
        default=None,
    )

    results_parser.set_defaults(func=results)
    results_parser.add_argument(
        "-d",
        "--data-path",
        help="Path to the root folder of all datasets.",
        type=Path,
        default=None,
    )
    results_parser.add_argument(
        "-o",
        "--output-path",
        help="Specifies the directory to save resulting tables.",
        type=Path,
        default=".",
    )

    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.unique_separator = "\t"
    args.best_ranker = 8
    args.best_ranker_params = {
        "rtype": 0,
        "srate": 0.5,
        "bag": 150,
        "frate": 0.3,
        "tree": 5,
        "leaf": 200,
        "shrinkage": 0.2,
    }
    args.func(args)


if __name__ == "__main__":
    main()
