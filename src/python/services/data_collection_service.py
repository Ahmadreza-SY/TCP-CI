import pandas as pd
from ..entities.execution_record import ExecutionRecord
from ..dataset_factory import DatasetFactory
from ..module_factory import ModuleFactory
import sys
from ..timer import tik, tok, save_time_measures
import subprocess
from ..execution_record_extractor.tr_torrent_processor import TrTorrentProcessor
import logging


class DataCollectionService:
    @staticmethod
    def fetch_source_code_if_needed(args):
        path = args.project_path
        slug = args.project_slug
        if path is None and slug is None:
            logging.error(
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
                logging.error("Failure in fetching source code for GitHub!")
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
            logging.error("No CI cycles found. Aborting...")
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
        save_time_measures(args.output_path, builds)

    @staticmethod
    def fetch_and_save_execution_history(args, repo_miner):
        execution_record_extractor = ModuleFactory.get_execution_record_extractor(
            args.language, args.ci_data_path
        )(args, repo_miner)
        exe_path = args.output_path / "exe.csv"
        exe_records, builds = execution_record_extractor.fetch_execution_records()
        exe_df = pd.DataFrame.from_records([e.to_dict() for e in exe_records])
        builds_df = pd.DataFrame.from_records([b.to_dict() for b in builds])
        if len(exe_df) > 0:
            exe_df.sort_values(
                by=[ExecutionRecord.BUILD, ExecutionRecord.JOB],
                ignore_index=True,
                inplace=True,
            )
            exe_df[ExecutionRecord.DURATION] = exe_df[ExecutionRecord.DURATION].abs()
            exe_df.to_csv(exe_path, index=False)
            builds_df.sort_values(
                by=["started_at"],
                ignore_index=True,
                inplace=True,
            )
            builds_df.to_csv(args.output_path / "builds.csv", index=False)
            return exe_records, builds
        else:
            logging.warn("No execution history collected!")
            return [], []

    @staticmethod
    def process_tr_torrent(args):
        processor = TrTorrentProcessor()
        processor.process_tr_torrent_data(args.input_path, args.output_path, args.repo)
