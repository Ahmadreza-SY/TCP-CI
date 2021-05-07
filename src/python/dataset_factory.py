from pydriller.git_repository import GitRepository
from .entities.entity import Entity
from .entities.entity_change import EntityChange
from .entities.execution_record import ExecutionRecord, TestVerdict
from tqdm import tqdm
import pandas as pd
from scipy.stats.mstats import gmean
import numpy as np

pd.options.mode.chained_assignment = None


class DatasetFactory:
    DEFAULT_VALUE = -1
    TEST = "Test"
    BUILD = "Build"
    # Complexity Metrics
    complexity_metrics = {
        "CountDeclFunction",
        "CountLine",
        "CountLineBlank",
        "CountLineCode",
        "CountLineCodeDecl",
        "CountLineCodeExe",
        "CountLineComment",
        "CountStmt",
        "CountStmtDecl",
        "CountStmtExe",
        "RatioCommentToCode",
        "MaxCyclomatic",
        "MaxCyclomaticModified",
        "MaxCyclomaticStrict",
        "MaxEssential",
        "MaxNesting",
        "SumCyclomatic",
        "SumCyclomaticModified",
        "SumCyclomaticStrict",
        "SumEssential",
        "CountDeclClass",
        "CountDeclClassMethod",
        "CountDeclClassVariable",
        "CountDeclExecutableUnit",
        "CountDeclInstanceMethod",
        "CountDeclInstanceVariable",
        "CountDeclMethod",
        "CountDeclMethodDefault",
        "CountDeclMethodPrivate",
        "CountDeclMethodProtected",
        "CountDeclMethodPublic",
    }
    # Process Metrics
    COMMIT_COUNT = "CommitCount"
    D_DEV_COUNT = "DistinctDevCount"
    LINES_ADDED = "LinesAdded"
    LINES_DELETED = "LinesDeleted"
    OWNERS_CONTRIBUTION = "OwnersContribution"
    MINOR_CONTRIBUTOR_COUNT = "MinorContributorCount"
    OWNERS_EXPERIENCE = "OwnersExperience"
    ALL_COMMITERS_EXPERIENCE = "AllCommitersExperience"
    process_metrics = {
        COMMIT_COUNT,
        D_DEV_COUNT,
        LINES_ADDED,
        LINES_DELETED,
        OWNERS_CONTRIBUTION,
        MINOR_CONTRIBUTOR_COUNT,
        OWNERS_EXPERIENCE,
        ALL_COMMITERS_EXPERIENCE,
    }
    # REC Features
    AVG_EXE_TIME = "AvgExeTime"
    MAX_EXE_TIME = "MaxExeTime"
    AGE = "Age"
    FAIL_RATE = "FailRate"
    ASSERT_RATE = "AssertRate"
    EXC_RATE = "ExcRate"
    LAST_VERDICT = "LastVerdict"
    LAST_EXE_TIME = "LastExeTime"
    rec_features = [
        AVG_EXE_TIME,
        MAX_EXE_TIME,
        AGE,
        FAIL_RATE,
        ASSERT_RATE,
        EXC_RATE,
        LAST_VERDICT,
        LAST_EXE_TIME,
    ]
    # COV Features
    CHN_SCORE_SUM = "ChnScoreSum"
    IMP_SCORE_SUM = "ImpScoreSum"
    CHN_COUNT = "ChnCount"
    IMP_COUNT = "ImpCount"
    # Ground Truth
    VERDICT = "Verdict"
    DURATION = "Duration"

    def __init__(
        self,
        config,
        change_history,
        repository_miner,
    ):
        self.config = config
        self.git_repository = GitRepository(config.project_path)
        self.change_history = change_history
        self.repository_miner = repository_miner

    def compute_static_metrics(self, ent_ids, ent_dict):
        metrics = {}
        for ent_id in ent_ids:
            metrics[ent_id] = {}
            for name, value in ent_dict[ent_id].items():
                if name in DatasetFactory.complexity_metrics:
                    metrics[ent_id][name] = value
        return metrics

    def compute_contributions(self, change_history):
        change_history["AuthoredLines"] = (
            change_history[EntityChange.ADDED_LINES]
            + change_history[EntityChange.DELETED_LINES]
        )
        devs = (
            change_history[[EntityChange.CONTRIBUTOR, "AuthoredLines"]]
            .groupby(EntityChange.CONTRIBUTOR, as_index=False)
            .sum()
            .sort_values("AuthoredLines", ascending=False, ignore_index=True)
        )
        devs["Exp"] = devs["AuthoredLines"] / devs["AuthoredLines"].sum() * 100.0
        return devs

    def compute_process_metrics(self, commit_hash, ent_ids, ent_dict):
        metrics = {}
        commit = self.git_repository.get_commit(commit_hash)
        commit_date = commit.author_date
        build_change_history = self.change_history[
            self.change_history[EntityChange.COMMIT_DATE] <= commit_date
        ]
        search_column = EntityChange.COMMIT
        if commit.merge:
            search_column = EntityChange.MERGE_COMMIT
        build_changes = self.change_history[
            self.change_history[search_column] == commit_hash
        ]
        project_devs = self.compute_contributions(build_change_history)
        for ent_id in ent_ids:
            metrics[ent_id] = {}
            ent_change_history = build_change_history[
                build_change_history[EntityChange.ID] == ent_id
            ]
            ent_changes = build_changes[build_changes[EntityChange.ID] == ent_id]
            ent_devs = self.compute_contributions(ent_change_history)
            ent_devs_ids = ent_devs[EntityChange.CONTRIBUTOR].values
            owner_id = ent_devs.iloc[0][EntityChange.CONTRIBUTOR]

            commit_count = len(ent_change_history)
            distict_dev_count = ent_change_history[EntityChange.CONTRIBUTOR].nunique()
            lines_added = ent_changes[EntityChange.ADDED_LINES].sum()
            lines_deleted = ent_changes[EntityChange.DELETED_LINES].sum()
            owners_contribution = ent_devs.iloc[0]["Exp"]
            minor_contributor_count = len(ent_devs[ent_devs["Exp"] < 5.0])
            owners_experience = project_devs[
                project_devs[EntityChange.CONTRIBUTOR] == owner_id
            ]["Exp"].values[0]
            ent_devs_exp = project_devs[
                project_devs[EntityChange.CONTRIBUTOR].isin(ent_devs_ids)
            ]["Exp"].values
            all_commiters_experience = gmean(ent_devs_exp)

            metrics[ent_id][DatasetFactory.COMMIT_COUNT] = commit_count
            metrics[ent_id][DatasetFactory.D_DEV_COUNT] = distict_dev_count
            metrics[ent_id][DatasetFactory.LINES_ADDED] = lines_added
            metrics[ent_id][DatasetFactory.LINES_DELETED] = lines_deleted
            metrics[ent_id][DatasetFactory.OWNERS_CONTRIBUTION] = owners_contribution
            metrics[ent_id][
                DatasetFactory.MINOR_CONTRIBUTOR_COUNT
            ] = minor_contributor_count
            metrics[ent_id][DatasetFactory.OWNERS_EXPERIENCE] = owners_experience
            metrics[ent_id][
                DatasetFactory.ALL_COMMITERS_EXPERIENCE
            ] = all_commiters_experience
        return metrics

    def compute_com_features(self, commit_hash, test_ids, ent_dict, build_tc_features):
        test_com_metrics = self.compute_static_metrics(test_ids, ent_dict)
        test_process_metrics = self.compute_process_metrics(
            commit_hash, test_ids, ent_dict
        )
        for test_id in test_ids:
            build_tc_features.setdefault(test_id, {})
            for name, value in test_com_metrics[test_id].items():
                build_tc_features[test_id][f"COM_{name}"] = value
            for name, value in test_process_metrics[test_id].items():
                build_tc_features[test_id][f"COM_{name}"] = value
        return build_tc_features

    def compute_rec_features(self, test_entities, exe_df, build, build_tc_features):
        exe_df.sort_values(by=[ExecutionRecord.BUILD], inplace=True, ignore_index=True)
        window = self.config.build_window
        builds = exe_df[ExecutionRecord.BUILD].unique().tolist()

        for test in test_entities.to_dict("records"):
            test_id = test[Entity.ID]
            build_tc_features.setdefault(test_id, {})
            build_result = exe_df[
                (exe_df[ExecutionRecord.BUILD] == build.id)
                & (exe_df[ExecutionRecord.TEST] == test_id)
            ].iloc[0]
            build_tc_features[test_id][DatasetFactory.VERDICT] = build_result[
                ExecutionRecord.VERDICT
            ]
            build_tc_features[test_id][DatasetFactory.DURATION] = build_result[
                ExecutionRecord.DURATION
            ]
            test_exe_df = exe_df[exe_df[ExecutionRecord.TEST] == test_id]
            test_exe_window = test_exe_df[test_exe_df[ExecutionRecord.BUILD] < build.id]
            if test_exe_window.empty:
                for feature in DatasetFactory.rec_features:
                    build_tc_features[test_id][
                        f"REC_{feature}"
                    ] = DatasetFactory.DEFAULT_VALUE
                build_tc_features[test_id][f"REC_{DatasetFactory.AGE}"] = 0
                continue

            if len(test_exe_window) > window:
                test_exe_window = test_exe_window.tail(window)
            first_build_id = test_exe_df.iloc[0][ExecutionRecord.BUILD]
            last_build_id = build_result[ExecutionRecord.BUILD]

            avg_exe_time = test_exe_window[ExecutionRecord.DURATION].mean()
            max_exe_time = test_exe_window[ExecutionRecord.DURATION].max()
            age = builds.index(last_build_id) - builds.index(first_build_id)
            fail_rate = len(
                test_exe_window[
                    test_exe_window[ExecutionRecord.VERDICT]
                    != TestVerdict.SUCCESS.value
                ]
            ) / len(test_exe_window)
            assert_rate = len(
                test_exe_window[
                    test_exe_window[ExecutionRecord.VERDICT]
                    == TestVerdict.ASSERTION.value
                ]
            ) / len(test_exe_window)
            exc_rate = len(
                test_exe_window[
                    test_exe_window[ExecutionRecord.VERDICT]
                    == TestVerdict.EXCEPTION.value
                ]
            ) / len(test_exe_window)
            last_verdict = test_exe_window.tail(1)[ExecutionRecord.VERDICT].values[0]
            last_exe_time = test_exe_window.tail(1)[ExecutionRecord.DURATION].values[0]

            build_tc_features[test_id][
                f"REC_{DatasetFactory.AVG_EXE_TIME}"
            ] = avg_exe_time
            build_tc_features[test_id][
                f"REC_{DatasetFactory.MAX_EXE_TIME}"
            ] = max_exe_time
            build_tc_features[test_id][f"REC_{DatasetFactory.AGE}"] = age
            build_tc_features[test_id][f"REC_{DatasetFactory.FAIL_RATE}"] = fail_rate
            build_tc_features[test_id][
                f"REC_{DatasetFactory.ASSERT_RATE}"
            ] = assert_rate
            build_tc_features[test_id][f"REC_{DatasetFactory.EXC_RATE}"] = exc_rate
            build_tc_features[test_id][
                f"REC_{DatasetFactory.LAST_VERDICT}"
            ] = last_verdict
            build_tc_features[test_id][
                f"REC_{DatasetFactory.LAST_EXE_TIME}"
            ] = last_exe_time
        return build_tc_features

    def compute_test_coverage(self, commit_hash, test_ids, src_ids):
        dep_graph, tar_graph = self.repository_miner.analyze_commit_dependency(
            commit_hash, test_ids, src_ids
        )
        build_commit = self.git_repository.get_commit(commit_hash)
        changed_ents = self.repository_miner.get_changed_entities(build_commit)
        impacted_ents = set()
        for changed_ent in changed_ents:
            impacted_ents.update(dep_graph.get_dependencies(changed_ent))

        coverage = {}
        for test_id in test_ids:
            coverage[test_id] = {"chn": [], "imp": []}
            target_ents = tar_graph.get_dependencies(test_id)
            target_weights = tar_graph.get_dependency_weights(test_id)
            for i, target in enumerate(target_ents):
                cov_score = 0.0 if target_weights[i] == 0 else target_weights[i][1]
                if target in changed_ents:
                    coverage[test_id]["chn"].append((target, cov_score))
                elif target in impacted_ents:
                    coverage[test_id]["imp"].append((target, cov_score))
        return coverage

    def compute_cov_features(self, test_ids, coverage, build_tc_features):
        for test_id in test_ids:
            build_tc_features.setdefault(test_id, {})
            changed_scores = [c[1] for c in coverage[test_id]["chn"]]
            impacted_scores = [c[1] for c in coverage[test_id]["imp"]]

            build_tc_features[test_id][f"COV_{DatasetFactory.CHN_SCORE_SUM}"] = sum(
                changed_scores
            )
            build_tc_features[test_id][f"COV_{DatasetFactory.IMP_SCORE_SUM}"] = sum(
                impacted_scores
            )
            build_tc_features[test_id][f"COV_{DatasetFactory.CHN_COUNT}"] = len(
                changed_scores
            )
            build_tc_features[test_id][f"COV_{DatasetFactory.IMP_COUNT}"] = len(
                impacted_scores
            )
        return build_tc_features

    def aggregate_cod_cov_metrics(self, covered_ents, com_metrics, process_metrics):
        scores = []
        metrics = {}
        for ent_id, score in covered_ents:
            scores.append(score)
            for name, value in com_metrics[ent_id].items():
                metrics.setdefault(name, [])
                metrics[name].append(value)
            for name, value in process_metrics[ent_id].items():
                metrics.setdefault(name, [])
                metrics[name].append(value)

        result = {}
        for name, values in metrics.items():
            weighted_sum = 0.0
            if sum(scores) == 0.0:
                weighted_sum = sum(values)
            else:
                v = np.array(values)
                w = np.array(scores)
                weighted_sum = v.dot(w / np.sum(w))
            result[name] = weighted_sum
        return result

    def compute_cod_cov_features(
        self, commit_hash, test_ids, coverage, ent_dict, build_tc_features
    ):
        all_affected_ents = set()
        for test_id in test_ids:
            all_affected_ents.update([c[0] for c in coverage[test_id]["chn"]])
            all_affected_ents.update([c[0] for c in coverage[test_id]["imp"]])

        all_affected_ents = list(all_affected_ents)
        ents_com_metrics = self.compute_static_metrics(all_affected_ents, ent_dict)
        ents_process_metrics = self.compute_process_metrics(
            commit_hash, all_affected_ents, ent_dict
        )

        for test_id in test_ids:
            build_tc_features.setdefault(test_id, {})
            for metric in (
                DatasetFactory.complexity_metrics | DatasetFactory.process_metrics
            ):
                build_tc_features[test_id][
                    f"COD_COV_CHN_{metric}"
                ] = DatasetFactory.DEFAULT_VALUE
            for metric in (
                DatasetFactory.complexity_metrics | DatasetFactory.process_metrics
            ):
                build_tc_features[test_id][
                    f"COD_COV_IMP_{metric}"
                ] = DatasetFactory.DEFAULT_VALUE

            changed_coverage = coverage[test_id]["chn"]
            agg_changed_metrics = self.aggregate_cod_cov_metrics(
                changed_coverage, ents_com_metrics, ents_process_metrics
            )
            impacted_coverage = coverage[test_id]["imp"]
            agg_impacted_metrics = self.aggregate_cod_cov_metrics(
                impacted_coverage, ents_com_metrics, ents_process_metrics
            )

            for name, value in agg_changed_metrics.items():
                build_tc_features[test_id][f"COD_COV_CHN_{name}"] = value
            for name, value in agg_impacted_metrics.items():
                build_tc_features[test_id][f"COD_COV_IMP_{name}"] = value

        return build_tc_features

    def select_valid_builds(self, builds, exe_df):
        builds.sort(key=lambda e: e.id)
        valid_builds = []
        for build in builds:
            metadata_path = (
                self.repository_miner.get_analysis_path(build.commit_hash)
                / "metadata.csv"
            )
            if not metadata_path.exists():
                result = self.repository_miner.analyze_commit_statically(
                    build.commit_hash
                )
                if result.empty:
                    continue
            build_exe_df = exe_df[exe_df[ExecutionRecord.BUILD] == build.id]
            if build_exe_df.empty:
                continue
            valid_builds.append(build)
        return valid_builds

    def create_dataset(self, builds, exe_records):
        exe_df = pd.DataFrame.from_records([e.to_dict() for e in exe_records])
        dataset = []
        valid_builds = self.select_valid_builds(builds, exe_df)
        for build in tqdm(valid_builds[1:], desc="Creating dataset"):
            commit_hash = build.commit_hash
            metadata_path = (
                self.repository_miner.get_analysis_path(commit_hash) / "metadata.csv"
            )
            entities_df = pd.read_csv(metadata_path)
            entities_dict = {e[Entity.ID]: e for e in entities_df.to_dict("records")}
            build_exe_df = exe_df[exe_df[ExecutionRecord.BUILD] == build.id]
            if build_exe_df.empty:
                continue
            test_ids = build_exe_df[ExecutionRecord.TEST].values.tolist()
            src_ids = list(set(entities_df[Entity.ID].values.tolist()) - set(test_ids))
            tests_df = entities_df[entities_df[Entity.ID].isin(test_ids)]

            build_tc_features = {}
            self.compute_com_features(
                commit_hash, test_ids, entities_dict, build_tc_features
            )
            self.compute_rec_features(tests_df, exe_df, build, build_tc_features)

            coverage = self.compute_test_coverage(commit_hash, test_ids, src_ids)
            self.compute_cov_features(test_ids, coverage, build_tc_features)
            self.compute_cod_cov_features(
                commit_hash, test_ids, coverage, entities_dict, build_tc_features
            )

            for test_id, features in build_tc_features.items():
                features[DatasetFactory.BUILD] = build.id
                features[DatasetFactory.TEST] = test_id
                dataset.append(features)
        return dataset

    def create_and_save_dataset(self, builds, exe_records):
        dataset = self.create_dataset(builds, exe_records)
        dataset_df = pd.DataFrame.from_records(dataset)
        cols = dataset_df.columns.tolist()
        cols.remove(DatasetFactory.BUILD)
        cols.remove(DatasetFactory.TEST)
        cols.insert(0, DatasetFactory.TEST)
        cols.insert(0, DatasetFactory.BUILD)
        dataset_df = dataset_df[cols]
        dataset_df.to_csv(self.config.output_path / "dataset.csv", index=False)
        print(f'Saved dataset to {self.config.output_path / "dataset.csv"}')