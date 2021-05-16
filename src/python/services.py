import pandas as pd
from .entities.execution_record import ExecutionRecord
from .dataset_factory import DatasetFactory
from .module_factory import ModuleFactory
import sys


class DataCollectionService:
    @staticmethod
    def create_dataset(args):
        repo_miner_class = ModuleFactory.get_repository_miner(args.level)
        repo_miner = repo_miner_class(args)
        change_history_df = repo_miner.compute_and_save_entity_change_history()

        records, builds = DataCollectionService.fetch_and_save_execution_history(
            args, repo_miner
        )

        if len(builds) == 0:
            print("No CI cycles found. Aborting...")
            sys.exit()

        dataset_factory = DatasetFactory(
            args,
            change_history_df,
            repo_miner,
        )
        dataset_factory.create_and_save_dataset(builds, records)

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
            return None, None
