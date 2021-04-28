from pydriller.git_repository import GitRepository
from .module_factory import ModuleFactory
from .entities.entity import EntityType, Entity
from .entities.entity_change import EntityChange
from tqdm import tqdm
import pandas as pd
from scipy.stats.mstats import gmean
from git import Git

pd.options.mode.chained_assignment = None


class DatasetFactory:
    TEST = "Test"
    BUILD = "Build"
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
    COMMIT_COUNT = "CommitCount"
    D_DEV_COUNT = "DistinctDevCount"
    LINES_ADDED = "LinesAdded"
    LINES_DELETED = "LinesDeleted"
    OWNERS_CONTRIBUTION = "OwnersContribution"
    MINOR_CONTRIBUTOR_COUNT = "MinorContributorCount"
    OWNERS_EXPERIENCE = "OwnersExperience"
    ALL_COMMITERS_EXPERIENCE = "AllCommitersExperience"

    def __init__(
        self, project_path, test_path, output_path, language, level, change_history
    ):
        self.project_path = project_path
        self.test_path = test_path
        self.output_path = output_path
        self.language = language
        self.level = level
        self.git_repository = GitRepository(project_path)
        self.change_history = change_history

    def compute_test_static_metrics(self, entities_df, build_tc_features):
        test_entities = entities_df[
            entities_df[Entity.ENTITY_TYPE] == EntityType.TEST.name
        ]
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
        g = Git(str(self.project_path))
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

    def compute_test_process_metrics(self, commit_hash, entities_df, build_tc_features):
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
        test_entities = entities_df[
            entities_df[Entity.ENTITY_TYPE] == EntityType.TEST.name
        ]
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

    def create_dataset(self, builds):
        builds.sort(key=lambda e: e.id)
        dataset = []
        for build in tqdm(builds, desc="Creating dataset"):
            metadata_path = (
                self.output_path / "metadata" / build.commit_hash / "metadata.csv"
            )
            entities_df = None
            if not metadata_path.exists():
                try:
                    self.git_repository.repo.git.checkout(build.commit_hash)
                except:
                    continue
                code_analyzer = ModuleFactory.get_code_analyzer(self.level)
                with code_analyzer(
                    self.project_path,
                    self.test_path,
                    self.output_path,
                    self.language,
                    self.level,
                ) as analyzer:
                    entities = analyzer.get_entities()
                    entities_df = pd.DataFrame.from_records(
                        [e.to_dict() for e in entities]
                    )
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    entities_df.to_csv(metadata_path, index=False)
            else:
                entities_df = pd.read_csv(metadata_path)

            build_tc_features = {}
            self.compute_test_static_metrics(entities_df, build_tc_features)
            self.compute_test_process_metrics(
                build.commit_hash, entities_df, build_tc_features
            )

            for test_id, features in build_tc_features.items():
                features[DatasetFactory.BUILD] = build.id
                features[DatasetFactory.TEST] = test_id
                dataset.append(features)
        return dataset

    def create_and_save_dataset(self, builds):
        dataset = self.create_dataset(builds)
        dataset_df = pd.DataFrame.from_records(dataset)
        cols = dataset_df.columns.tolist()
        cols.remove(DatasetFactory.BUILD)
        cols.remove(DatasetFactory.TEST)
        cols.insert(0, DatasetFactory.TEST)
        cols.insert(0, DatasetFactory.BUILD)
        dataset_df = dataset_df[cols]
        dataset_df.to_csv(self.output_path / "dataset.csv", index=False)
        print(f'Saved dataset to {self.output_path / "dataset.csv"}')