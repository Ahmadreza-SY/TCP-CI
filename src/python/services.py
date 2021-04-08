from .commit_miner import *
from .release_feature_extractor import *
from .association_miner import FileAssociationMiner, FunctionAssociationMiner
import pandas as pd
from os.path import isfile
import os
import json
from .entities.entity import Entity, EntityType
from .code_analyzer.code_analyzer import AnalysisLevel
from .entities.execution_record import ExecutionRecord


class DataCollectionService:
    @staticmethod
    def compute_and_save_dep_data(code_analyzer, metadata_df, args):
        output_path = args.output_path
        if (
            (output_path / "metadata.csv").exists()
            and (output_path / "dep.csv").exists()
            and (output_path / "tar.csv").exists()
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
            str(args.project_path), since=args.since, only_in_branch=args.branch
        )
        miner = miner_type(repository, metadata_df, args.language)
        dep_graph = miner.compute_dependency_weights(dep_graph)
        tar_graph = miner.compute_dependency_weights(tar_graph)
        dep_graph.save_graph(
            args.output_path / "dep.csv", "dependencies", args.unique_separator
        )
        tar_graph.reverse_graph()
        tar_graph.save_graph(
            args.output_path / "tar.csv", "targeted_by_tests", args.unique_separator
        )

    @staticmethod
    def compute_and_save_commit_data(args):
        output_path = args.output_path
        if (output_path / "commits.csv").exists() and (
            output_path / "contributors.csv"
        ).exists():
            print(f"Commmit datasets already exist, skipping commit mining.")
            return
        commit_miner = CommitMiner(args.project_path)
        commit_features, contributors = commit_miner.mine_commits()
        commit_features.to_csv(
            args.output_path / "commits.csv", index=False, sep=args.unique_separator
        )
        contributors.to_csv(args.output_path / "contributors.csv", index=False)

    @staticmethod
    def compute_and_save_historical_data(args, code_analyzer):
        entities = code_analyzer.get_entities()
        metadata_df = pd.DataFrame.from_records([e.to_dict() for e in entities])
        metadata_cols = metadata_df.columns.values.tolist()
        metadata_df.to_csv(
            args.output_path / "metadata.csv", index=False, columns=metadata_cols
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
    def compute_test_case_features(exe_df):
        def calc_failure_rate(tc_results, row):
            tc_runs = tc_results[(row[ExecutionRecord.TEST])]
            tc_passes = 0 if 0 not in tc_runs.index else tc_runs[0]
            return 1 - tc_passes / tc_runs.sum()

        test_cases = pd.DataFrame()
        tc_age = exe_df.groupby(ExecutionRecord.TEST, as_index=False).count()
        test_cases[ExecutionRecord.TEST] = tc_age[ExecutionRecord.TEST]
        test_cases["age"] = tc_age[ExecutionRecord.VERDICT]
        tc_avg_duration = (
            exe_df[[ExecutionRecord.TEST, ExecutionRecord.DURATION]]
            .groupby(ExecutionRecord.TEST, as_index=False)
            .mean()
        )
        test_cases = pd.merge(test_cases, tc_avg_duration, on=[ExecutionRecord.TEST])
        tc_results = (
            exe_df[[ExecutionRecord.TEST, ExecutionRecord.VERDICT]]
            .groupby([ExecutionRecord.TEST, ExecutionRecord.VERDICT], as_index=False)
            .size()
        )
        test_cases["failure_rate"] = test_cases.apply(
            lambda r: calc_failure_rate(tc_results, r), axis=1
        )
        test_cases.rename(
            columns={ExecutionRecord.DURATION: "avg_duration"}, inplace=True
        )
        return test_cases

    @staticmethod
    def compute_contributors_failure_rate(
        exe_df, builds_df, commits_df, contributors_df
    ):
        def update_contirbutor_failure_rate(fr_map, contributor, result):
            if contributor not in fr_map:
                fr_map[contributor] = [0, 0]
            if result > 0:
                fr_map[contributor][1] += 1
            else:
                fr_map[contributor][0] += 1

        builds_df.rename(columns={"id": ExecutionRecord.BUILD}, inplace=True)
        builds_tc_results = exe_df.groupby(ExecutionRecord.BUILD, as_index=False).sum()
        builds_tc_results = pd.merge(
            builds_tc_results, builds_df, on=[ExecutionRecord.BUILD]
        )
        commit_to_build_result = dict(
            zip(
                builds_tc_results.commit_hash,
                builds_tc_results[ExecutionRecord.VERDICT],
            )
        )
        contributor_failure_rate = {}
        for index, commit in commits_df.iterrows():
            if commit.hash not in commit_to_build_result:
                continue
            if commit.committer == commit.author:
                update_contirbutor_failure_rate(
                    contributor_failure_rate,
                    commit.committer,
                    commit_to_build_result[commit.hash],
                )
            else:
                update_contirbutor_failure_rate(
                    contributor_failure_rate,
                    commit.committer,
                    commit_to_build_result[commit.hash],
                )
                update_contirbutor_failure_rate(
                    contributor_failure_rate,
                    commit.author,
                    commit_to_build_result[commit.hash],
                )
        contributor_failure_rate = pd.DataFrame(
            {
                "id": list(contributor_failure_rate.keys()),
                "failure_rate": list(
                    map(lambda r: r[1] / sum(r), contributor_failure_rate.values())
                ),
            }
        )
        contributors_df = pd.merge(
            contributors_df, contributor_failure_rate, on=["id"], how="outer"
        )
        contributors_df.fillna(0, inplace=True)
        return contributors_df

    @staticmethod
    def fetch_and_save_execution_history(args, execution_record_extractor):
        exe_path = args.output_path / "exe.csv"
        if exe_path.exists() and exe_path.stat().st_size > 0:
            print(
                f"Skipping {args.project_slug} repository, execution history {exe_path} already exists."
            )
            return

        exe_records, builds = execution_record_extractor.fetch_execution_records()
        exe_df = pd.DataFrame.from_records([e.to_dict() for e in exe_records])
        if len(exe_df) > 0:
            exe_df.to_csv(exe_path, index=False)

            DataCollectionService.compute_test_case_features(exe_df).to_csv(
                args.output_path / "test_cases.csv", index=False
            )

            builds_df = pd.DataFrame.from_records([b.to_dict() for b in builds])
            commits_df = pd.read_csv(
                args.output_path / "commits.csv",
                usecols=["hash", "committer", "author"],
                sep=args.unique_separator,
            )
            contributors_df = pd.read_csv(args.output_path / "contributors.csv")
            contributors_df = DataCollectionService.compute_contributors_failure_rate(
                exe_df, builds_df, commits_df, contributors_df
            )
            contributors_df.to_csv(args.output_path / "contributors.csv", index=False)
        else:
            print("No execution history collected!")

    @staticmethod
    def compute_and_save_release_data(args):
        output_path = args.output_path
        metadata_df = pd.read_csv(args.histories_dir / "metadata.csv")

        miner_type = None
        if args.level == AnalysisLevel.FILE:
            miner_type = FileAssociationMiner
        elif args.level == AnalysisLevel.FUNCTION:
            miner_type = FunctionAssociationMiner
        repository = RepositoryMining(
            str(args.project_path),
            from_commit=args.from_commit,
            to_commit=args.to_commit,
        )
        miner = miner_type(repository, metadata_df, args.language)

        print("Extracting release impacts ...")
        changed_sets = miner.compute_changed_sets()
        changed_entities = set.union(*changed_sets) if len(changed_sets) > 0 else set()
        dep_graph = pd.read_csv(
            args.histories_dir / "dep.csv",
            sep=args.unique_separator,
            converters={"dependencies": json.loads, "weights": json.loads},
        )
        tar_graph = pd.read_csv(
            args.histories_dir / "tar.csv",
            sep=args.unique_separator,
            converters={"targeted_by_tests": json.loads, "weights": json.loads},
        )
        release_impacts = ReleaseFeatureExtractor.extract_release_impacts(
            changed_entities, dep_graph, tar_graph
        )
        release_impacts.to_csv(
            output_path / "release_impacts.csv", sep=args.unique_separator, index=False
        )

        print("Extracting release changes ...")
        release_changes = ReleaseFeatureExtractor.extract_release_changes(
            metadata_df, args.project_path, args.from_commit, args.to_commit
        )
        pd.DataFrame(release_changes).to_csv(
            output_path / "release_changes.csv", index=False
        )

        print(f"All finished, results are saved in {output_path}")
