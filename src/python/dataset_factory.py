from pydriller.domain.commit import DMMProperty
from .entities.entity import Entity
from .entities.entity_change import EntityChange
from .entities.execution_record import ExecutionRecord
from tqdm import tqdm
import pandas as pd
from scipy.stats.mstats import gmean
import numpy as np
from .timer import tik, tok, tik_list, tok_list
import sys
from .feature_extractor.feature import Feature
from .feature_extractor.rec_feature_extractor import RecFeatureExtractor
import logging


class DatasetFactory:
    def __init__(
        self,
        config,
        change_history,
        repository_miner,
    ):
        self.config = config
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
                if name in Feature.complexity_metrics:
                    metrics[ent_id][name] = value
        return metrics

    def compute_change_metrics(self, build, ent_ids):
        metrics = {}
        commits = self.repository_miner.get_build_commits(build)
        all_merge = all([commit.merge for commit in commits])
        modifications = []
        for commit in commits:
            if not all_merge and commit.merge:
                continue
            modifications.extend(self.repository_miner.compute_modifications(commit))

        mod_map = {}
        for mod in modifications:
            changed_entity_id = self.repository_miner.get_changed_entity_id(mod)
            if changed_entity_id not in mod_map:
                mod_map[changed_entity_id] = []
            mod_map[changed_entity_id].append(mod)

        for ent_id in ent_ids:
            if ent_id not in mod_map:
                continue

            ent_mods = mod_map[ent_id]
            metrics.setdefault(ent_id, {})
            metrics[ent_id][Feature.LINES_ADDED] = sum(
                [ent_mod.added for ent_mod in ent_mods]
            )
            metrics[ent_id][Feature.LINES_DELETED] = sum(
                [ent_mod.removed for ent_mod in ent_mods]
            )

            added_scatterings = []
            for ent_mod in ent_mods:
                diff = ent_mod.diff_parsed
                added_lines = list(map(lambda p: p[0], diff["added"]))
                added_scatterings.append(
                    self.repository_miner.compute_scattering(added_lines)
                )
            metrics[ent_id][Feature.ADDED_CHANGE_SCATTERING] = max(added_scatterings)

            deleted_scatterings = []
            for ent_mod in ent_mods:
                diff = ent_mod.diff_parsed
                deleted_lines = list(map(lambda p: p[0], diff["deleted"]))
                deleted_scatterings.append(
                    self.repository_miner.compute_scattering(deleted_lines)
                )
            metrics[ent_id][Feature.DELETED_CHANGE_SCATTERING] = max(
                deleted_scatterings
            )

            dmm_size = self.repository_miner.compute_dmm(
                ent_mods, DMMProperty.UNIT_SIZE
            )
            dmm_complexity = self.repository_miner.compute_dmm(
                ent_mods, DMMProperty.UNIT_COMPLEXITY
            )
            dmm_interfacing = self.repository_miner.compute_dmm(
                ent_mods, DMMProperty.UNIT_INTERFACING
            )
            if dmm_size is not None:
                metrics[ent_id][Feature.DMM_SIZE] = dmm_size
            if dmm_complexity is not None:
                metrics[ent_id][Feature.DMM_COMPLEXITY] = dmm_complexity
            if dmm_interfacing is not None:
                metrics[ent_id][Feature.DMM_INTERFACING] = dmm_interfacing
        return metrics

    def compute_process_metrics(self, build, ent_ids):
        metrics = {}
        commit = self.repository_miner.get_build_commits(build)[0]
        build_change_history = self.change_history[
            (self.change_history[EntityChange.COMMIT_DATE] <= commit.committer_date)
            & (self.change_history[EntityChange.MERGE_COMMIT] == False)
        ]
        project_devs = self.compute_contributions(build_change_history.copy())
        for ent_id in ent_ids:
            metrics.setdefault(ent_id, {})
            ent_change_history = build_change_history[
                build_change_history[EntityChange.ID] == ent_id
            ]
            authored_lines_sum = (
                ent_change_history[EntityChange.ADDED_LINES].sum()
                + ent_change_history[EntityChange.DELETED_LINES].sum()
            )
            if ent_change_history.empty or authored_lines_sum == 0:
                ent_change_history = self.change_history[
                    (
                        self.change_history[EntityChange.COMMIT_DATE]
                        <= commit.committer_date
                    )
                    & (self.change_history[EntityChange.ID] == ent_id)
                ]
            ent_devs = self.compute_contributions(ent_change_history.copy())
            ent_devs_ids = ent_devs[EntityChange.CONTRIBUTOR].values
            ent_devs_exp = project_devs[
                project_devs[EntityChange.CONTRIBUTOR].isin(ent_devs_ids)
            ]["Exp"].values
            owner_id = ent_devs.iloc[0][EntityChange.CONTRIBUTOR]

            metrics[ent_id][Feature.COMMIT_COUNT] = len(ent_change_history)
            metrics[ent_id][Feature.D_DEV_COUNT] = ent_change_history[
                EntityChange.CONTRIBUTOR
            ].nunique()
            metrics[ent_id][Feature.OWNERS_CONTRIBUTION] = ent_devs.iloc[0]["Exp"]
            metrics[ent_id][Feature.MINOR_CONTRIBUTOR_COUNT] = len(
                ent_devs[ent_devs["Exp"] < 5.0]
            )
            metrics[ent_id][Feature.OWNERS_EXPERIENCE] = project_devs[
                project_devs[EntityChange.CONTRIBUTOR] == owner_id
            ]["Exp"].values[0]
            metrics[ent_id][Feature.ALL_COMMITERS_EXPERIENCE] = gmean(ent_devs_exp)
        return metrics

    def compute_all_metrics(self, build, ent_ids, ent_dict, prefix="", chn_build=None):
        tik(f"{prefix}COM_M", build.id)
        static_metrics = self.compute_static_metrics(ent_ids, ent_dict)
        tok(f"{prefix}COM_M", build.id)
        tik(f"{prefix}CHN_M", build.id)
        change_metrics_build = build if chn_build is None else chn_build
        change_metrics = self.compute_change_metrics(change_metrics_build, ent_ids)
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
            for metric in Feature.all_metrics:
                prefix = Feature.get_metric_prefix(metric)
                build_tc_features[test_id][
                    f"TES_{prefix}_{metric}"
                ] = Feature.DEFAULT_VALUE
            for name, value in test_metrics[test_id].items():
                prefix = Feature.get_metric_prefix(name)
                build_tc_features[test_id][f"TES_{prefix}_{name}"] = value
        return build_tc_features

    def compute_test_coverage(self, build, test_ids, src_ids, chn_build=None):
        dep_graph, tar_graph = self.repository_miner.analyze_commit_dependency(
            build, test_ids, src_ids
        )
        chn_build = build if chn_build is None else chn_build
        chn_commits = self.repository_miner.get_build_commits(chn_build)
        modifications = []
        for commit in chn_commits:
            modifications.extend(self.repository_miner.compute_modifications(commit))

        changed_ents = set()
        for mod in modifications:
            changed_entity_id = self.repository_miner.get_changed_entity_id(mod)
            changed_ents.add(changed_entity_id)
        tik("Impacted", build.id)
        impacted_ents = set()
        for changed_ent in changed_ents:
            impacted_ents.update(dep_graph.get_dependencies(changed_ent))
        tok("Impacted", build.id)

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

    def compute_cov_features(self, build, test_ids, coverage, build_tc_features):
        for test_id in test_ids:
            build_tc_features.setdefault(test_id, {})
            if test_id in coverage:
                changed_scores = [c[1] for c in coverage[test_id]["chn"]]
                tik("Impacted", build.id)
                impacted_scores = [c[1] for c in coverage[test_id]["imp"]]
                tok("Impacted", build.id)
            else:
                changed_scores, impacted_scores = [], []

            build_tc_features[test_id][f"COV_{Feature.CHN_SCORE_SUM}"] = sum(
                changed_scores
            )
            build_tc_features[test_id][f"COV_{Feature.IMP_SCORE_SUM}"] = sum(
                impacted_scores
            )
            build_tc_features[test_id][f"COV_{Feature.CHN_COUNT}"] = len(changed_scores)
            build_tc_features[test_id][f"COV_{Feature.IMP_COUNT}"] = len(
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
            if test_id in coverage:
                all_affected_ents.update([c[0] for c in coverage[test_id]["chn"]])
                all_affected_ents.update([c[0] for c in coverage[test_id]["imp"]])
        return list(all_affected_ents)

    def compute_cod_cov_features(
        self, build, test_ids, coverage, ent_dict, build_tc_features, chn_build=None
    ):
        all_affected_ents = self.get_all_affected_ents(test_ids, coverage)
        ents_metrics = self.compute_all_metrics(
            build, all_affected_ents, ent_dict, "COD_COV_", chn_build
        )

        tik_list(["COD_COV_COM_M", "COD_COV_PRO_M", "COD_COV_CHN_M"], build.id)
        for test_id in test_ids:
            if test_id in coverage:
                changed_coverage = coverage[test_id]["chn"]
                agg_changed_metrics = self.aggregate_cov_metrics(
                    changed_coverage, ents_metrics
                )

                tik("Impacted", build.id)
                impacted_coverage = coverage[test_id]["imp"]
                agg_impacted_metrics = self.aggregate_cov_metrics(
                    impacted_coverage, ents_metrics
                )
                tok("Impacted", build.id)
            else:
                agg_changed_metrics = {}
                agg_impacted_metrics = {}

            build_tc_features.setdefault(test_id, {})
            for metric in Feature.all_metrics:
                prefix = Feature.get_metric_prefix(metric)
                build_tc_features[test_id][
                    f"COD_COV_{prefix}_C_{metric}"
                ] = Feature.DEFAULT_VALUE
            for name, value in agg_changed_metrics.items():
                prefix = Feature.get_metric_prefix(name)
                build_tc_features[test_id][f"COD_COV_{prefix}_C_{name}"] = value

            tik("Impacted", build.id)
            for metric in Feature.complexity_metrics + Feature.process_metrics:
                prefix = Feature.get_metric_prefix(metric)
                build_tc_features[test_id][
                    f"COD_COV_{prefix}_IMP_{metric}"
                ] = Feature.DEFAULT_VALUE
            for name, value in agg_impacted_metrics.items():
                prefix = Feature.get_metric_prefix(name)
                build_tc_features[test_id][f"COD_COV_{prefix}_IMP_{name}"] = value
            tok("Impacted", build.id)

        tok_list(["COD_COV_COM_M", "COD_COV_PRO_M", "COD_COV_CHN_M"], build.id)
        return build_tc_features

    def compute_det_features(self, build, test_ids, coverage, build_tc_features):
        commit = self.repository_miner.get_build_commits(build)[0]
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
            faults[ent_id][Feature.FAULTS] = ent_faults

        for test_id in test_ids:
            if test_id in coverage:
                changed_coverage = coverage[test_id]["chn"]
                agg_changed_metrics = self.aggregate_cov_metrics(
                    changed_coverage, faults
                )
                tik("Impacted", build.id)
                impacted_coverage = coverage[test_id]["imp"]
                agg_impacted_metrics = self.aggregate_cov_metrics(
                    impacted_coverage, faults
                )
                tok("Impacted", build.id)
            else:
                agg_changed_metrics = {}
                agg_impacted_metrics = {}

            build_tc_features.setdefault(test_id, {})
            build_tc_features[test_id][
                f"DET_COV_C_{Feature.FAULTS}"
            ] = Feature.DEFAULT_VALUE
            for name, value in agg_changed_metrics.items():
                build_tc_features[test_id][f"DET_COV_C_{name}"] = value

            tik("Impacted", build.id)
            build_tc_features[test_id][
                f"DET_COV_IMP_{Feature.FAULTS}"
            ] = Feature.DEFAULT_VALUE
            for name, value in agg_impacted_metrics.items():
                build_tc_features[test_id][f"DET_COV_IMP_{name}"] = value
            tok("Impacted", build.id)

        return build_tc_features

    def select_valid_builds(self, builds, exe_df):
        builds.sort(key=lambda e: e.started_at)
        valid_builds = []
        for build in builds:
            metadata_path = (
                self.repository_miner.get_analysis_path(build) / "metadata.csv"
            )
            if not metadata_path.exists():
                result = self.repository_miner.analyze_build_statically(build)
                if result.empty:
                    continue
            build_exe_df = exe_df[exe_df[ExecutionRecord.BUILD] == build.id]
            if build_exe_df.empty:
                continue
            valid_builds.append(build)
        return valid_builds

    def create_build_change_history(self, builds):
        build_ids = []
        commits = []
        for b in builds:
            for c in b.commits:
                build_ids.append(b.id)
                commits.append(c)
        builds_df = pd.DataFrame({"BuildId": build_ids, EntityChange.COMMIT: commits})
        build_change_history = self.change_history.merge(
            builds_df, on=EntityChange.COMMIT, how="left"
        )
        build_change_history["BuildId"] = (
            build_change_history["BuildId"].fillna(-1).astype(int)
        )
        return build_change_history

    def create_dataset(self, builds, exe_records):
        builds.sort(key=lambda b: b.started_at)
        exe_df = pd.DataFrame.from_records([e.to_dict() for e in exe_records])
        build_time_dict = dict(
            zip([b.id for b in builds], [b.started_at for b in builds])
        )
        exe_df["started_at"] = exe_df[ExecutionRecord.BUILD].apply(
            lambda b: build_time_dict[b]
        )
        exe_df.sort_values("started_at", ignore_index=True, inplace=True)
        exe_df.drop("started_at", inplace=True, axis=1)

        dataset = []
        valid_builds = self.select_valid_builds(builds, exe_df)
        failed_builds = set(
            exe_df[exe_df[ExecutionRecord.VERDICT] > 0][ExecutionRecord.BUILD]
            .unique()
            .tolist()
        )
        valid_builds = [b for b in valid_builds if b.id in failed_builds]
        build_change_history = self.create_build_change_history(builds)
        rec_extractor = RecFeatureExtractor(
            self.config.build_window, exe_df, build_change_history
        )

        if len(valid_builds) == 0:
            logging.error("No valid builds found. Aborting ...")
            sys.exit()

        for build in tqdm(valid_builds[1:], desc="Creating dataset"):
            tik("Total", build.id)
            metadata_path = (
                self.repository_miner.get_analysis_path(build) / "metadata.csv"
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
            rec_extractor.compute_rec_features(tests_df, build, build_tc_features)
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
            self.compute_cov_features(build, test_ids, coverage, build_tc_features)
            tok("COV_M", build.id)
            self.compute_cod_cov_features(
                build, test_ids, coverage, entities_dict, build_tc_features
            )
            tik("DET_COV_M", build.id)
            self.compute_det_features(build, test_ids, coverage, build_tc_features)
            tok("DET_COV_M", build.id)

            for test_id, features in build_tc_features.items():
                features[Feature.BUILD] = build.id
                features[Feature.TEST] = test_id
                dataset.append(features)
            tok("Total", build.id)
        return dataset

    def create_and_save_dataset(self, builds, exe_records):
        dataset = self.create_dataset(builds, exe_records)
        if len(dataset) == 0:
            logging.error("No dataset created!")
            return
        dataset_df = pd.DataFrame.from_records(dataset)
        cols = dataset_df.columns.tolist()
        cols.remove(Feature.BUILD)
        cols.remove(Feature.TEST)
        cols.insert(0, Feature.TEST)
        cols.insert(0, Feature.BUILD)
        dataset_df = dataset_df[cols]
        dataset_df[Feature.DURATION] = dataset_df[Feature.DURATION].abs()
        dataset_df.to_csv(self.config.output_path / "dataset.csv", index=False)
        self.repository_miner.clean_analysis_path()
        logging.info(f'Saved dataset to {self.config.output_path / "dataset.csv"}')
