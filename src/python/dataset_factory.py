from pydriller.git_repository import GitRepository
from pydriller import RepositoryMining
from .module_factory import ModuleFactory
from .entities.entity import EntityType, Entity
from .entities.entity_change import EntityChange
from .entities.execution_record import ExecutionRecord, TestVerdict
from tqdm import tqdm
import pandas as pd
from scipy.stats.mstats import gmean
from git import Git

pd.options.mode.chained_assignment = None


class DatasetFactory:
    TEST = "Test"
    BUILD = "Build"
    # Complexity Metrics
    COM_static_feature_set = {
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
    # REC Features
    AVG_EXE_TIME = "AvgExeTime"
    MAX_EXE_TIME = "MaxExeTime"
    AGE = "Age"
    FAIL_RATE = "FailRate"
    ASSERT_RATE = "AssertRate"
    EXC_RATE = "ExcRate"
    LAST_VERDICT = "LastVerdict"
    LAST_EXE_TIME = "LastExeTime"
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

    def compute_static_metrics(self, test_entities, build_tc_features):
        for test in test_entities.to_dict("records"):
            for name, value in test.items():
                if name in DatasetFactory.COM_static_feature_set:
                    test_id = test[Entity.ID]
                    build_tc_features.setdefault(test_id, {})
                    build_tc_features[test_id][f"COM_{name}"] = value
        return build_tc_features

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

    def get_merge_commits(self, merge_commit):
        if not merge_commit.merge:
            return [merge_commit]

        parents = merge_commit.parents
        g = Git(str(self.config.project_path))
        merge_base = g.execute(f"git merge-base {parents[0]} {parents[1]}".split())
        merge_commits = []
        last_commit = self.git_repository.get_commit(parents[1])
        while last_commit.hash != merge_base:
            if last_commit.merge:
                merge_commits.extend(self.get_merge_commits(last_commit))
            else:
                merge_commits.append(last_commit)
            last_commit = self.git_repository.get_commit(last_commit.parents[0])
        return merge_commits

    def compute_process_metrics(self, commit_hash, test_entities, build_tc_features):
        commit = self.git_repository.get_commit(commit_hash)
        commit_date = commit.author_date
        build_change_history = self.change_history[
            self.change_history[EntityChange.COMMIT_DATE] <= commit_date
        ]
        commit_hashes = [commit_hash]
        if commit.merge:
            commit_hashes = map(lambda c: c.hash, self.get_merge_commits(commit))
        commit_changes = self.change_history[
            self.change_history[EntityChange.COMMIT].isin(commit_hashes)
        ]
        project_devs = self.compute_contributions(build_change_history)
        for test in test_entities.to_dict("records"):
            test_id = test[Entity.ID]
            build_tc_features.setdefault(test_id, {})
            test_change_history = build_change_history[
                build_change_history[EntityChange.ID] == test_id
            ]
            test_changes = commit_changes[commit_changes[EntityChange.ID] == test_id]
            test_devs = self.compute_contributions(test_change_history)
            test_devs_ids = test_devs[EntityChange.CONTRIBUTOR].values
            owner_id = test_devs.iloc[0][EntityChange.CONTRIBUTOR]

            commit_count = len(test_change_history)
            distict_dev_count = test_change_history[EntityChange.CONTRIBUTOR].nunique()
            lines_added = test_changes[EntityChange.ADDED_LINES].sum()
            lines_deleted = test_changes[EntityChange.DELETED_LINES].sum()
            owners_contribution = test_devs.iloc[0]["Exp"]
            minor_contributor_count = len(test_devs[test_devs["Exp"] < 5.0])
            owners_experience = project_devs[
                project_devs[EntityChange.CONTRIBUTOR] == owner_id
            ]["Exp"].values[0]
            test_devs_exp = project_devs[
                project_devs[EntityChange.CONTRIBUTOR].isin(test_devs_ids)
            ]["Exp"].values
            all_commiters_experience = gmean(test_devs_exp)

            build_tc_features[test_id][
                f"COM_{DatasetFactory.COMMIT_COUNT}"
            ] = commit_count
            build_tc_features[test_id][
                f"COM_{DatasetFactory.D_DEV_COUNT}"
            ] = distict_dev_count
            build_tc_features[test_id][
                f"COM_{DatasetFactory.LINES_ADDED}"
            ] = lines_added
            build_tc_features[test_id][
                f"COM_{DatasetFactory.LINES_DELETED}"
            ] = lines_deleted
            build_tc_features[test_id][
                f"COM_{DatasetFactory.OWNERS_CONTRIBUTION}"
            ] = owners_contribution
            build_tc_features[test_id][
                f"COM_{DatasetFactory.MINOR_CONTRIBUTOR_COUNT}"
            ] = minor_contributor_count
            build_tc_features[test_id][
                f"COM_{DatasetFactory.OWNERS_EXPERIENCE}"
            ] = owners_experience
            build_tc_features[test_id][
                f"COM_{DatasetFactory.ALL_COMMITERS_EXPERIENCE}"
            ] = all_commiters_experience
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

    def compute_co_changes(self, commit_hash, ent_ids_set):
        from .services import DataCollectionService

        DataCollectionService.checkout_default_branch(self.config.project_path)
        build_commit = self.git_repository.get_commit(commit_hash)
        commit_date = build_commit.author_date
        build_change_history = self.change_history[
            self.change_history[EntityChange.COMMIT_DATE] <= commit_date
        ]
        commit_change_lists = (
            build_change_history[[EntityChange.ID, EntityChange.COMMIT]]
            .groupby(EntityChange.COMMIT)[EntityChange.ID]
            .apply(list)
            .reset_index(name="changes")
        )
        commit_change_lists_d = dict(
            zip(
                commit_change_lists[EntityChange.COMMIT].values.tolist(),
                commit_change_lists["changes"].values.tolist(),
            )
        )
        repository = RepositoryMining(str(self.config.project_path), to=commit_date)
        co_changes = []
        for commit in repository.traverse_commits():
            changes = set()
            commit_hashes = [commit.hash]
            if commit.merge:
                commit_hashes = map(lambda c: c.hash, self.get_merge_commits(commit))
            for change_hash in commit_hashes:
                changes.update(commit_change_lists_d[change_hash])
            entity_changes = changes.intersection(ent_ids_set)
            if len(entity_changes) > 0:
                co_changes.append(entity_changes)
        return co_changes

    def compute_cov_features(self, commit_hash, test_ids, src_ids):
        co_changes = self.compute_co_changes(commit_hash, set(test_ids) | set(src_ids))
        dep_graph, tar_graph = self.repository_miner.analyze_commit_dependency(
            commit_hash, test_ids, src_ids, co_changes
        )

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
            build_exe_df = exe_df[exe_df[ExecutionRecord.BUILD] == build.id]
            if build_exe_df.empty:
                continue
            test_ids = build_exe_df[ExecutionRecord.TEST].values.tolist()
            src_ids = list(set(entities_df[Entity.ID].values.tolist()) - set(test_ids))
            tests_df = entities_df[entities_df[Entity.ID].isin(test_ids)]

            build_tc_features = {}
            self.compute_static_metrics(tests_df, build_tc_features)
            self.compute_process_metrics(commit_hash, tests_df, build_tc_features)
            self.compute_rec_features(tests_df, exe_df, build, build_tc_features)

            self.compute_cov_features(commit_hash, test_ids, src_ids)

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