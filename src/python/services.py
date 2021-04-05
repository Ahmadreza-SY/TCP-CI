from .exe_feature_extractor import *
from .commit_miner import *
from .release_feature_extractor import *
from .association_miner import FileAssociationMiner, FunctionAssociationMiner
import pandas as pd
from os.path import isfile
import json
from .entities.entity import Entity, EntityType
from .code_analyzer.code_analyzer import AnalysisLevel


class DataCollectionService:
    @staticmethod
    def compute_and_save_dep_data(code_analyzer, metadata_df, args):
        output_path = args.output_path
        if (
            isfile(f"{output_path}/metadata.csv")
            and isfile(f"{output_path}/dep.csv")
            and isfile(f"{output_path}/tar.csv")
        ):
            print(f"Dependency datasets already exist, skipping dependency analysis.")
            return

        src_ids = metadata_df[
            metadata_df[Entity.ENTITY_TYPE] == EntityType.SRC.name
        ].Id.values
        test_ids = metadata_df[
            metadata_df[Entity.ENTITY_TYPE] == EntityType.TEST.name
        ].Id.values
        dep_graph = code_analyzer.compute_dependency_graph(src_ids, src_ids)
        tar_graph = code_analyzer.compute_dependency_graph(test_ids, src_ids)

        miner_type = None
        if args.level == AnalysisLevel.FILE:
            miner_type = FileAssociationMiner
        elif args.level == AnalysisLevel.FUNCTION:
            miner_type = FunctionAssociationMiner

        repository = RepositoryMining(
            args.project_path, since=args.since, only_in_branch=args.branch
        )
        miner = miner_type(repository, metadata_df, args.language)
        dep_graph = miner.compute_dependency_weights(dep_graph)
        tar_graph = miner.compute_dependency_weights(tar_graph)
        dep_graph.save_graph(
            f"{args.output_path}/dep.csv", "dependencies", args.unique_separator
        )
        tar_graph.reverse_graph()
        tar_graph.save_graph(
            f"{args.output_path}/tar.csv", "targeted_by_tests", args.unique_separator
        )

    @staticmethod
    def compute_and_save_commit_data(args):
        output_path = args.output_path
        if isfile(f"{output_path}/commits.csv") and isfile(
            f"{output_path}/contributors.csv"
        ):
            print(f"Commmit datasets already exist, skipping commit mining.")
            return
        commit_miner = CommitMiner(args.project_path)
        commit_features, contributors = commit_miner.mine_commits()
        commit_features.to_csv(
            f"{args.output_path}/commits.csv", index=False, sep=args.unique_separator
        )
        contributors.to_csv(f"{args.output_path}/contributors.csv", index=False)

    @staticmethod
    def compute_and_save_historical_data(args, code_analyzer):
        entities = code_analyzer.get_entities()
        metadata_df = pd.DataFrame.from_records([e.to_dict() for e in entities])
        metadata_cols = metadata_df.columns.values.tolist()
        metadata_df.to_csv(
            f"{args.output_path}/metadata.csv", index=False, columns=metadata_cols
        )
        test_count = len(
            metadata_df[metadata_df[Entity.ENTITY_TYPE] == EntityType.TEST.name]
        )
        print(
            f"Found a total of {len(metadata_df)} entities including {test_count} among test code and {len(metadata_df) - test_count} among the main source code."
        )

        DataCollectionService.compute_and_save_dep_data(
            code_analyzer, metadata_df, args
        )
        DataCollectionService.compute_and_save_commit_data(args)

    @staticmethod
    def fetch_and_save_execution_history(args):
        ExeFeatureExtractor.fetch_and_save_execution_history(args)

    @staticmethod
    def compute_and_save_release_data(args):
        output_path = args.output_path
        metadata_df = pd.read_csv(f"{args.histories_dir}/metadata.csv")

        miner_type = None
        if args.level == AnalysisLevel.FILE:
            miner_type = FileAssociationMiner
        elif args.level == AnalysisLevel.FUNCTION:
            miner_type = FunctionAssociationMiner
        repository = RepositoryMining(
            args.project_path, from_commit=args.from_commit, to_commit=args.to_commit
        )
        miner = miner_type(repository, metadata_df, args.language)

        print("Extracting release impacts ...")
        changed_sets = miner.compute_changed_sets()
        changed_entities = set.union(*changed_sets) if len(changed_sets) > 0 else set()
        dep_graph = pd.read_csv(
            f"{args.histories_dir}/dep.csv",
            sep=args.unique_separator,
            converters={"dependencies": json.loads, "weights": json.loads},
        )
        tar_graph = pd.read_csv(
            f"{args.histories_dir}/tar.csv",
            sep=args.unique_separator,
            converters={"targeted_by_tests": json.loads, "weights": json.loads},
        )
        release_impacts = ReleaseFeatureExtractor.extract_release_impacts(
            changed_entities, dep_graph, tar_graph
        )
        release_impacts.to_csv(
            f"{output_path}/release_impacts.csv", sep=args.unique_separator, index=False
        )

        print("Extracting release changes ...")
        release_changes = ReleaseFeatureExtractor.extract_release_changes(
            metadata_df, args.project_path, args.from_commit, args.to_commit
        )
        pd.DataFrame(release_changes).to_csv(
            f"{output_path}/release_changes.csv", index=False
        )

        print(f"All finished, results are saved in {output_path}")
