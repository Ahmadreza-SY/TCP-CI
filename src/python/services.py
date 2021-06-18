import pandas as pd
from .entities.execution_record import ExecutionRecord
from .dataset_factory import DatasetFactory
from .module_factory import ModuleFactory
import sys
from .timer import tik, tok, save_time_measures
from .ranklib_learner import RankLibLearner
import subprocess
from .constants import *


class DataCollectionService:
    @staticmethod
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
            clone_command = (
                f"git clone https://github.com/{slug}.git {args.project_path}"
            )
            return_code = subprocess.call(clone_command, shell=True)
            if return_code != 0:
                print("Failure in fetching source code for GitHub!")
                sys.exit()

    @staticmethod
    def create_dataset(args):
        DataCollectionService.fetch_source_code_if_needed(args)
        tik("Dataset Collection")
        repo_miner_class = ModuleFactory.get_repository_miner(args.level)
        repo_miner = repo_miner_class(args)
        tik("Change History")
        change_history_df = repo_miner.compute_and_save_entity_change_history()
        tok("Change History")
        tik("Execution History")
        records, builds = DataCollectionService.fetch_and_save_execution_history(
            args, repo_miner
        )
        tok("Execution History")
        if len(builds) == 0:
            print("No CI cycles found. Aborting...")
            sys.exit()

        tik("Feature Extraction")
        dataset_factory = DatasetFactory(
            args,
            change_history_df,
            repo_miner,
        )
        dataset_factory.create_and_save_dataset(builds, records)
        tok("Feature Extraction")
        tok("Dataset Collection")
        save_time_measures(args.output_path)

    @staticmethod
    def fetch_and_save_execution_history(args, repo_miner):
        execution_record_extractor = ModuleFactory.get_execution_record_extractor(
            args.language
        )(args, repo_miner)
        exe_path = args.output_path / "exe.csv"
        exe_records, builds = execution_record_extractor.fetch_execution_records()
        exe_df = pd.DataFrame.from_records([e.to_dict() for e in exe_records])
        if len(exe_df) > 0:
            exe_df.sort_values(
                by=[ExecutionRecord.BUILD, ExecutionRecord.JOB],
                ignore_index=True,
                inplace=True,
            )
            exe_df.to_csv(exe_path, index=False)
            return exe_records, builds
        else:
            print("No execution history collected!")
            return [], []

    @staticmethod
    def run_all_tsp_accuracy_experiments(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            print("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        print(f"##### Running experiments for {dataset_path.parent.name} #####")
        learner = RankLibLearner(args)
        dataset_df = pd.read_csv(dataset_path)
        results_path = args.output_path / "tsp_accuracy_results"
        print("***** Running full feature set experiments *****")
        learner.run_accuracy_experiments(dataset_df, "full", results_path)
        print()
        print("***** Running w/o impacted feature set experiments *****")
        learner.run_accuracy_experiments(
            dataset_df.drop(IMPACTED_FEATURES, axis=1), "wo-impacted", results_path
        )
        print()
        feature_groups_names = {
            "TES_COM": TES_COM,
            "TES_PRO": TES_PRO,
            "TES_CHN": TES_CHN,
            "REC": REC,
            "COV": COV,
            "COD_COV_COM": COD_COV_COM,
            "COD_COV_PRO": COD_COV_PRO,
            "COD_COV_CHN": COD_COV_CHN,
            "DET_COV": DET_COV,
        }
        for feature_group, names in feature_groups_names.items():
            print(f"***** Running w/o {feature_group} feature set experiments *****")
            learner.run_accuracy_experiments(
                dataset_df.drop(names, axis=1), f"wo-{feature_group}", results_path
            )
            print()
