import pandas as pd
from ..decay_dataset_factory import DecayDatasetFactory
from ..dataset_factory import DatasetFactory
from ..feature_extractor.feature import Feature
from ..module_factory import ModuleFactory
import sys
from ..ranklib_learner import RankLibLearner
from ..feature_extractor.feature import Feature
from ..code_analyzer.code_analyzer import AnalysisLevel
from ..results.results_analyzer import ResultAnalyzer


class ExperimentsService:
    @staticmethod
    def run_all_tsp_accuracy_experiments(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            print("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        print(f"##### Running experiments for {dataset_path.parent.name} #####")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= args.test_count:
            print(
                f"Not enough builds for training: require at least {args.test_count + 1}, found {builds_count}"
            )
            sys.exit()
        results_path = args.output_path / "tsp_accuracy_results"
        print("***** Running full feature set experiments *****")
        learner.run_accuracy_experiments(dataset_df, "full", results_path)
        learner.test_heuristics(dataset_df, results_path / "full")
        print()
        print("***** Running w/o impacted feature set experiments *****")
        learner.run_accuracy_experiments(
            dataset_df.drop(Feature.IMPACTED_FEATURES, axis=1),
            "wo-impacted",
            results_path,
        )
        print()
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
        for feature_group, names in feature_groups_names.items():
            print(f"***** Running w/o {feature_group} feature set experiments *****")
            learner.run_accuracy_experiments(
                dataset_df.drop(names, axis=1), f"wo-{feature_group}", results_path
            )
            print()

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
        for feature_group, names in high_level_feature_groups.items():
            print(f"***** Running with {feature_group} feature set experiments *****")
            learner.run_accuracy_experiments(
                dataset_df[non_feature_cols + names], f"W-{feature_group}", results_path
            )
            print()

    @staticmethod
    def run_decay_test_experiments(args):
        print(f"Running decay tests for {args.output_path.name}")
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
        models_path = args.output_path / "tsp_accuracy_results" / "full"
        decay_ds_factory.create_decay_datasets(dataset_df, models_path)

        learner = RankLibLearner(args)
        datasets_path = args.output_path / "decay_datasets"
        learner.run_decay_test_experiments(datasets_path, models_path)
        print(f"All finished and results are saved at {datasets_path}")
        print()

    @staticmethod
    def analyze_results(args):
        result_analyzer = ResultAnalyzer(args)
        result_analyzer.analyze_results()
