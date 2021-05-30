from pydriller.git_repository import GitRepository
from pydriller.domain.commit import DMMProperty
from .entities.entity import Entity
from .entities.entity_change import EntityChange
from .entities.execution_record import ExecutionRecord, TestVerdict
from tqdm import tqdm
import pandas as pd
from scipy.stats.mstats import gmean
import numpy as np
from .timer import tik, tok, tik_list, tok_list
import sys

pd.options.mode.chained_assignment = None


class DatasetFactory:
    DEFAULT_VALUE = -1
    TEST = "Test"
    BUILD = "Build"
    # Complexity Metrics
    complexity_metrics = [
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
    ]
    # Process Metrics
    COMMIT_COUNT = "CommitCount"
    D_DEV_COUNT = "DistinctDevCount"
    OWNERS_CONTRIBUTION = "OwnersContribution"
    MINOR_CONTRIBUTOR_COUNT = "MinorContributorCount"
    OWNERS_EXPERIENCE = "OwnersExperience"
    ALL_COMMITERS_EXPERIENCE = "AllCommitersExperience"
    process_metrics = [
        COMMIT_COUNT,
        D_DEV_COUNT,
        OWNERS_CONTRIBUTION,
        MINOR_CONTRIBUTOR_COUNT,
        OWNERS_EXPERIENCE,
        ALL_COMMITERS_EXPERIENCE,
    ]
    # Change Metrics
    LINES_ADDED = "LinesAdded"
    LINES_DELETED = "LinesDeleted"
    ADDED_CHANGE_SCATTERING = "AddedChangeScattering"
    DELETED_CHANGE_SCATTERING = "DeletedChangeScattering"
    DMM_SIZE = "DMMSize"
    DMM_COMPLEXITY = "DMMComplexity"
    DMM_INTERFACING = "DMMInterfacing"
    change_metrics = [
        LINES_ADDED,
        LINES_DELETED,
        ADDED_CHANGE_SCATTERING,
        DELETED_CHANGE_SCATTERING,
        DMM_SIZE,
        DMM_COMPLEXITY,
        DMM_INTERFACING,
    ]
    # All Metrics
    all_metrics = complexity_metrics + process_metrics + change_metrics
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
    # DET Features
    FAULTS = "Faults"
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

    def compute_static_metrics(self, ent_ids, ent_dict):
        metrics = {}
        for ent_id in ent_ids:
            metrics.setdefault(ent_id, {})
            for name, value in ent_dict[ent_id].items():
                if name in DatasetFactory.complexity_metrics:
                    metrics[ent_id][name] = value
        return metrics

    def compute_change_metrics(self, build, ent_ids):
        metrics = {}
        commit = self.git_repository.get_commit(build.commit_hash)
        modifications = self.repository_miner.compute_modifications(commit)
        mod_map = {}
        for mod in modifications:
            changed_entity_id = self.repository_miner.get_changed_entity_id(mod)
            mod_map[changed_entity_id] = mod

        for ent_id in ent_ids:
            if ent_id not in mod_map:
                continue

            ent_mod = mod_map[ent_id]
            metrics.setdefault(ent_id, {})
            metrics[ent_id][DatasetFactory.LINES_ADDED] = ent_mod.added
            metrics[ent_id][DatasetFactory.LINES_DELETED] = ent_mod.removed
            diff = ent_mod.diff_parsed
            added_lines = list(map(lambda p: p[0], diff["added"]))
            metrics[ent_id][
                DatasetFactory.ADDED_CHANGE_SCATTERING
            ] = self.repository_miner.compute_scattering(added_lines)
            deleted_lines = list(map(lambda p: p[0], diff["deleted"]))
            metrics[ent_id][
                DatasetFactory.DELETED_CHANGE_SCATTERING
            ] = self.repository_miner.compute_scattering(deleted_lines)

            dmm_size = self.repository_miner.compute_dmm(ent_mod, DMMProperty.UNIT_SIZE)
            dmm_complexity = self.repository_miner.compute_dmm(
                ent_mod, DMMProperty.UNIT_COMPLEXITY
            )
            dmm_interfacing = self.repository_miner.compute_dmm(
                ent_mod, DMMProperty.UNIT_INTERFACING
            )
            if dmm_size is not None:
                metrics[ent_id][DatasetFactory.DMM_SIZE] = dmm_size
            if dmm_complexity is not None:
                metrics[ent_id][DatasetFactory.DMM_COMPLEXITY] = dmm_complexity
            if dmm_interfacing is not None:
                metrics[ent_id][DatasetFactory.DMM_INTERFACING] = dmm_interfacing
        return metrics

    def compute_process_metrics(self, build, ent_ids):
        metrics = {}
        commit = self.git_repository.get_commit(build.commit_hash)
        build_change_history = self.change_history[
            (self.change_history[EntityChange.COMMIT_DATE] <= commit.committer_date)
            & (self.change_history[EntityChange.MERGE_COMMIT] == False)
        ]
        project_devs = self.compute_contributions(build_change_history)
        for ent_id in ent_ids:
            metrics.setdefault(ent_id, {})
            ent_change_history = build_change_history[
                build_change_history[EntityChange.ID] == ent_id
            ]
            if ent_change_history.empty:
                ent_change_history = self.change_history[
                    (
                        self.change_history[EntityChange.COMMIT_DATE]
                        <= commit.committer_date
                    )
                    & (self.change_history[EntityChange.ID] == ent_id)
                ]
            ent_devs = self.compute_contributions(ent_change_history)
            ent_devs_ids = ent_devs[EntityChange.CONTRIBUTOR].values
            ent_devs_exp = project_devs[
                project_devs[EntityChange.CONTRIBUTOR].isin(ent_devs_ids)
            ]["Exp"].values
            owner_id = ent_devs.iloc[0][EntityChange.CONTRIBUTOR]

            metrics[ent_id][DatasetFactory.COMMIT_COUNT] = len(ent_change_history)
            metrics[ent_id][DatasetFactory.D_DEV_COUNT] = ent_change_history[
                EntityChange.CONTRIBUTOR
            ].nunique()
            metrics[ent_id][DatasetFactory.OWNERS_CONTRIBUTION] = ent_devs.iloc[0][
                "Exp"
            ]
            metrics[ent_id][DatasetFactory.MINOR_CONTRIBUTOR_COUNT] = len(
                ent_devs[ent_devs["Exp"] < 5.0]
            )
            metrics[ent_id][DatasetFactory.OWNERS_EXPERIENCE] = project_devs[
                project_devs[EntityChange.CONTRIBUTOR] == owner_id
            ]["Exp"].values[0]
            metrics[ent_id][DatasetFactory.ALL_COMMITERS_EXPERIENCE] = gmean(
                ent_devs_exp
            )
        return metrics

    def compute_all_metrics(self, build, ent_ids, ent_dict, prefix=""):
        tik(f"{prefix}COM_M", build.id)
        static_metrics = self.compute_static_metrics(ent_ids, ent_dict)
        tok(f"{prefix}COM_M", build.id)
        tik(f"{prefix}CHN_M", build.id)
        change_metrics = self.compute_change_metrics(build, ent_ids)
        tok(f"{prefix}CHN_M", build.id)
        tik(f"{prefix}PRO_M", build.id)
        process_metrics = self.compute_process_metrics(build, ent_ids)
        tok(f"{prefix}PRO_M", build.id)
        computed_metrics = [static_metrics, change_metrics, process_metrics]
        all_metrics = {}
        for computed_metric in computed_metrics:
            for ent_id, metrics in computed_metric.items():
                all_metrics.setdefault(ent_id, {})
                for name, value in metrics.items():
                    all_metrics[ent_id][name] = value
        return all_metrics

    def compute_tes_features(self, build, test_ids, ent_dict, build_tc_features):
        test_metrics = self.compute_all_metrics(build, test_ids, ent_dict, "TES_")
        for test_id in test_ids:
            build_tc_features.setdefault(test_id, {})
            for metric in DatasetFactory.all_metrics:
                build_tc_features[test_id][
                    f"TES_{metric}"
                ] = DatasetFactory.DEFAULT_VALUE
            for name, value in test_metrics[test_id].items():
                build_tc_features[test_id][f"TES_{name}"] = value
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

    def compute_test_coverage(self, build, test_ids, src_ids):
        dep_graph, tar_graph = self.repository_miner.analyze_commit_dependency(
            build, test_ids, src_ids
        )
        build_commit = self.git_repository.get_commit(build.commit_hash)
        modifications = self.repository_miner.compute_modifications(build_commit)
        changed_ents = set()
        for mod in modifications:
            changed_entity_id = self.repository_miner.get_changed_entity_id(mod)
            changed_ents.add(changed_entity_id)
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

    def aggregate_cov_metrics(self, covered_ents, metrics):
        metrics_list = {}
        for ent_id, score in covered_ents:
            for name, value in metrics[ent_id].items():
                metrics_list.setdefault(name, [])
                metrics_list[name].append((value, score))

        result = {}
        for name, values in metrics_list.items():
            metric_values = [v[0] for v in values]
            scores = [v[1] for v in values]
            weighted_sum = 0.0
            if sum(scores) == 0.0:
                weighted_sum = sum(metric_values)
            else:
                v = np.array(metric_values)
                w = np.array(scores)
                weighted_sum = v.dot(w / np.sum(w))
            result[name] = weighted_sum
        return result

    def get_all_affected_ents(self, test_ids, coverage):
        all_affected_ents = set()
        for test_id in test_ids:
            all_affected_ents.update([c[0] for c in coverage[test_id]["chn"]])
            all_affected_ents.update([c[0] for c in coverage[test_id]["imp"]])
        return list(all_affected_ents)

    def compute_cod_cov_features(
        self, build, test_ids, coverage, ent_dict, build_tc_features
    ):
        all_affected_ents = self.get_all_affected_ents(test_ids, coverage)
        ents_metrics = self.compute_all_metrics(
            build, all_affected_ents, ent_dict, "COD_COV_"
        )

        tik_list(["COD_COV_COM_M", "COD_COV_PRO_M", "COD_COV_CHN_M"], build.id)
        for test_id in test_ids:
            changed_coverage = coverage[test_id]["chn"]
            agg_changed_metrics = self.aggregate_cov_metrics(
                changed_coverage, ents_metrics
            )
            impacted_coverage = coverage[test_id]["imp"]
            agg_impacted_metrics = self.aggregate_cov_metrics(
                impacted_coverage, ents_metrics
            )

            build_tc_features.setdefault(test_id, {})
            for metric in DatasetFactory.all_metrics:
                build_tc_features[test_id][
                    f"COD_COV_CHN_{metric}"
                ] = DatasetFactory.DEFAULT_VALUE
            for metric in (
                DatasetFactory.complexity_metrics + DatasetFactory.process_metrics
            ):
                build_tc_features[test_id][
                    f"COD_COV_IMP_{metric}"
                ] = DatasetFactory.DEFAULT_VALUE

            for name, value in agg_changed_metrics.items():
                build_tc_features[test_id][f"COD_COV_CHN_{name}"] = value
            for name, value in agg_impacted_metrics.items():
                build_tc_features[test_id][f"COD_COV_IMP_{name}"] = value
        tok_list(["COD_COV_COM_M", "COD_COV_PRO_M", "COD_COV_CHN_M"], build.id)
        return build_tc_features

    def compute_det_features(self, build, test_ids, coverage, build_tc_features):
        commit = self.git_repository.get_commit(build.commit_hash)
        build_change_history = self.change_history[
            (self.change_history[EntityChange.COMMIT_DATE] <= commit.committer_date)
            & (self.change_history[EntityChange.MERGE_COMMIT] == False)
        ]
        all_affected_ents = self.get_all_affected_ents(test_ids, coverage)
        faults = {}
        for ent_id in all_affected_ents:
            ent_faults = build_change_history[
                build_change_history[EntityChange.ID] == ent_id
            ][EntityChange.BUG_FIX].sum()
            faults[ent_id] = {}
            faults[ent_id][DatasetFactory.FAULTS] = ent_faults

        for test_id in test_ids:
            changed_coverage = coverage[test_id]["chn"]
            agg_changed_metrics = self.aggregate_cov_metrics(changed_coverage, faults)
            impacted_coverage = coverage[test_id]["imp"]
            agg_impacted_metrics = self.aggregate_cov_metrics(impacted_coverage, faults)

            build_tc_features.setdefault(test_id, {})
            build_tc_features[test_id][
                f"DET_COV_CHN_{DatasetFactory.FAULTS}"
            ] = DatasetFactory.DEFAULT_VALUE
            build_tc_features[test_id][
                f"DET_COV_IMP_{DatasetFactory.FAULTS}"
            ] = DatasetFactory.DEFAULT_VALUE

            for name, value in agg_changed_metrics.items():
                build_tc_features[test_id][f"DET_COV_CHN_{name}"] = value
            for name, value in agg_impacted_metrics.items():
                build_tc_features[test_id][f"DET_COV_IMP_{name}"] = value

        return build_tc_features

    def select_valid_builds(self, builds, exe_df):
        all_commits_set = set([c.hash for c in self.repository_miner.get_all_commits()])
        builds.sort(key=lambda e: e.id)
        valid_builds = []
        for build in builds:
            if build.commit_hash not in all_commits_set:
                continue
            metadata_path = (
                self.repository_miner.get_analysis_path(build.commit_hash)
                / "metadata.csv"
            )
            if not metadata_path.exists():
                result = self.repository_miner.analyze_commit_statically(build)
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
        valid_build_ids = [b.id for b in valid_builds]
        exe_df = exe_df[exe_df[ExecutionRecord.BUILD].isin(valid_build_ids)]

        if len(valid_builds) == 0:
            print("No valid builds found. Aborting ...")
            sys.exit()

        for build in tqdm(valid_builds[1:], desc="Creating dataset"):
            metadata_path = (
                self.repository_miner.get_analysis_path(build.commit_hash)
                / "metadata.csv"
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
            self.compute_tes_features(build, test_ids, entities_dict, build_tc_features)
            tik("REC_M", build.id)
            self.compute_rec_features(tests_df, exe_df, build, build_tc_features)
            tok("REC_M", build.id)

            tik_list(
                [
                    "COV_P",
                    "COD_COV_COM_P",
                    "COD_COV_PRO_P",
                    "COD_COV_CHN_P",
                    "DET_COV_P",
                ],
                build.id,
            )
            coverage = self.compute_test_coverage(build, test_ids, src_ids)
            tok_list(
                [
                    "COV_P",
                    "COD_COV_COM_P",
                    "COD_COV_PRO_P",
                    "COD_COV_CHN_P",
                    "DET_COV_P",
                ],
                build.id,
            )
            tik("COV_M", build.id)
            self.compute_cov_features(test_ids, coverage, build_tc_features)
            tok("COV_M", build.id)
            self.compute_cod_cov_features(
                build, test_ids, coverage, entities_dict, build_tc_features
            )
            tik("DET_COV_M", build.id)
            self.compute_det_features(build, test_ids, coverage, build_tc_features)
            tok("DET_COV_M", build.id)

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