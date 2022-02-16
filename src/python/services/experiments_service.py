import pandas as pd
from ..decay_dataset_factory import DecayDatasetFactory
from ..dataset_factory import DatasetFactory
from ..feature_extractor.feature import Feature
from ..module_factory import ModuleFactory
import sys
from ..ranklib_learner import RankLibLearner
from ..code_analyzer.code_analyzer import AnalysisLevel
from ..results.results_analyzer import ResultAnalyzer
from ..hyp_param_opt import HypParamOpt
from .data_service import DataService
from pathlib import Path
from enum import Enum
import logging


class Experiment(Enum):
    FULL = "FULL"
    WO_IMP = "WO_IMP"
    WO_TES_COM = "WO_TES_COM"
    WO_TES_PRO = "WO_TES_PRO"
    WO_TES_CHN = "WO_TES_CHN"
    WO_REC = "WO_REC"
    WO_COV = "WO_COV"
    WO_COD_COV_COM = "WO_COD_COV_COM"
    WO_COD_COV_PRO = "WO_COD_COV_PRO"
    WO_COD_COV_CHN = "WO_COD_COV_CHN"
    WO_DET_COV = "WO_DET_COV"
    W_Code = "W_Code"
    W_Execution = "W_Execution"
    W_Coverage = "W_Coverage"


class ExperimentsService:
    @staticmethod
    def run_best_ranker_experiments(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            logging.error("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        logging.info("Reading the dataset.")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            logging.error(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        results_path = args.output_path / "tsp_accuracy_results"
        outliers_dataset_df = DataService.remove_outlier_tests(
            args.output_path, dataset_df
        )
        logging.info("Finished reading the dataset.")

        logging.info(
            f"***** Running {args.experiment.value} experiment for {dataset_path.parent.name} *****"
        )
        if args.experiment == Experiment.FULL:
            learner.run_accuracy_experiments(
                outliers_dataset_df, "full-outliers", results_path
            )
            learner.test_heuristics(outliers_dataset_df, results_path / "full-outliers")
        elif args.experiment == Experiment.WO_IMP:
            learner.run_accuracy_experiments(
                outliers_dataset_df.drop(Feature.IMPACTED_FEATURES, axis=1),
                "wo-impacted-outliers",
                results_path,
            )
        elif (
            args.experiment.value.startswith("WO_")
            and args.experiment != Experiment.WO_IMP
        ):
            feature_groups_names = {
                "TES_COM": Feature.TES_COM,
                "TES_PRO": Feature.TES_PRO,
                "TES_CHN": Feature.TES_CHN,
                "REC": Feature.REC,
                "COV": Feature.COV,
                "COD_COV_COM": Feature.COD_COV_COM,
                "COD_COV_PRO": Feature.COD_COV_PRO,
                "COD_COV_CHN": Feature.COD_COV_CHN,
                "DET_COV": Feature.DET_COV,
            }
            feature_group = args.experiment.value[3:]
            names = feature_groups_names[feature_group]
            learner.run_accuracy_experiments(
                outliers_dataset_df.drop(names, axis=1),
                f"wo-{feature_group}-outliers",
                results_path,
            )
        elif args.experiment.value.startswith("W_"):
            test_code_features = Feature.TES_COM + Feature.TES_PRO + Feature.TES_CHN
            test_execution_features = Feature.REC
            test_coverage_features = (
                Feature.COV
                + Feature.COD_COV_COM
                + Feature.COD_COV_PRO
                + Feature.COD_COV_CHN
                + Feature.DET_COV
            )
            high_level_feature_groups = {
                "Code": test_code_features,
                "Execution": test_execution_features,
                "Coverage": test_coverage_features,
            }
            non_feature_cols = [
                Feature.BUILD,
                Feature.TEST,
                Feature.VERDICT,
                Feature.DURATION,
            ]
            feature_group = args.experiment.value[2:]
            names = high_level_feature_groups[feature_group]
            learner.run_accuracy_experiments(
                outliers_dataset_df[non_feature_cols + names],
                f"W-{feature_group}-outliers",
                results_path,
            )
        logging.info("Done run_best_ranker_experiments")

    @staticmethod
    def run_all_tcp_rankers(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            logging.error("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        logging.info(f"##### Running experiments for {dataset_path.parent.name} #####")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            logging.error(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        outliers_dataset_df = DataService.remove_outlier_tests(
            args.output_path, dataset_df
        )
        rankers = {
            0: ("MART", {"tree": 30}),
            6: (
                "LambdaMART",
                {"tree": 30, "metric2T": "NDCG@10", "metric2t": "NDCG@10"},
            ),
            2: ("RankBoost", {}),
            4: ("CoordinateAscent", {}),
            7: ("ListNet", {}),
            8: ("RandomForest", {}),
        }
        results_path = args.output_path / "tcp_rankers"
        for id, info in rankers.items():
            name, params = info
            logging.info(
                f"***** Running {name} full feature set without Outliers experiments *****"
            )
            learner.run_accuracy_experiments(
                outliers_dataset_df, name, results_path, ranker=(id, params)
            )

    @staticmethod
    def run_decay_test_experiments(args):
        logging.info(f"Running decay tests for {args.output_path.name}")
        repo_miner_class = ModuleFactory.get_repository_miner(AnalysisLevel.FILE)
        repo_miner = repo_miner_class(args)
        change_history_df = repo_miner.load_entity_change_history()
        dataset_factory = DatasetFactory(
            args,
            change_history_df,
            repo_miner,
        )
        dataset_df = pd.read_csv(args.output_path / "dataset.csv")
        decay_ds_factory = DecayDatasetFactory(dataset_factory, args)
        models_path = args.output_path / "tsp_accuracy_results" / "full-outliers"
        decay_ds_factory.create_decay_datasets(dataset_df, models_path)

        learner = RankLibLearner(args)
        datasets_path = args.output_path / "decay_datasets"
        learner.run_decay_test_experiments(datasets_path, models_path)
        logging.info(f"All finished and results are saved at {datasets_path}")
        print()

    @staticmethod
    def analyze_results(args):
        result_analyzer = ResultAnalyzer(args)
        result_analyzer.analyze_results()

    @staticmethod
    def hyp_param_opt(args):
        optimizer = HypParamOpt(args)
        logging.info(f"***** Running {args.output_path.name} hypopt *****")
        build_ds_path = Path(args.output_path / "hyp_param_opt" / str(args.build))
        optimizer.run_optimization(build_ds_path, args.comb_index)
