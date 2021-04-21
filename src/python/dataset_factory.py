from pydriller.git_repository import GitRepository
from .module_factory import ModuleFactory
from .entities.entity import EntityType, Entity
from tqdm import tqdm
import pandas as pd


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

    def __init__(self, project_path, test_path, output_path, language, level):
        self.project_path = project_path
        self.test_path = test_path
        self.output_path = output_path
        self.language = language
        self.level = level
        self.git_repository = GitRepository(project_path)

    def compute_test_static_metrics(self, entities_df, build_tc_features):
        for entity in entities_df.to_dict("records"):
            if entity[Entity.ENTITY_TYPE] == EntityType.TEST.name:
                for name, value in entity.items():
                    if name in DatasetFactory.COM_static_feature_set:
                        entity_id = entity[Entity.ID]
                        build_tc_features.setdefault(entity_id, {})
                        build_tc_features[entity_id][f"COM_{name}"] = value
        return build_tc_features

    def create_dataset(self, builds):
        builds.sort(key=lambda e: e.id)
        dataset = []
        for build in tqdm(builds, desc="Creating dataset"):
            metadata_path = (
                self.output_path / "metadata" / build.commit_hash / "metadata.csv"
            )
            if not metadata_path.exists():
                continue

            build_tc_features = {}
            entities_df = pd.read_csv(metadata_path)
            self.compute_test_static_metrics(entities_df, build_tc_features)

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