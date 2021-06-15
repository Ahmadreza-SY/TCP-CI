import pandas as pd
from .entities.execution_record import ExecutionRecord
from .dataset_factory import DatasetFactory
from .module_factory import ModuleFactory
import sys
from .timer import tik, tok, save_time_measures
from .ranklib_learner import RankLibLearner
import subprocess


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
    def learn(args):
        dataset_path = args.output_path / "dataset.csv"
        if not dataset_path.exists():
            print("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        learner = RankLibLearner(args)

        print("Starting NRPA experiments")
        nrpa_dataset = pd.read_csv(dataset_path)
        nrpa_ranklib_ds = learner.convert_to_ranklib_dataset(nrpa_dataset)
        traning_sets_path = args.output_path / "nrpa-full"
        learner.create_ranklib_training_sets(nrpa_ranklib_ds, traning_sets_path)
        nrpa_results = learner.train_and_test("nrpa", traning_sets_path)
        nrpa_results.to_csv(traning_sets_path / "results.csv", index=False)

        failed_builds = (
            nrpa_dataset[nrpa_dataset[DatasetFactory.VERDICT] > 0][DatasetFactory.BUILD]
            .unique()
            .tolist()
        )
        if len(failed_builds) > 1:
            print("Starting APFD experiments")
            apfd_dataset = nrpa_dataset[
                nrpa_dataset[DatasetFactory.BUILD].isin(failed_builds)
            ].reset_index()
            apfd_ranklib_ds = learner.convert_to_ranklib_dataset(apfd_dataset)
            traning_sets_path = args.output_path / "apfd-full"
            learner.create_ranklib_training_sets(apfd_ranklib_ds, traning_sets_path)
            apfd_results = learner.train_and_test("napfd", traning_sets_path)
            apfd_results.to_csv(traning_sets_path / "results.csv", index=False)
        else:
            print("Not enough failed builds for APFD experiments.")
