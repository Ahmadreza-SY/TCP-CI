from pydriller.git_repository import GitRepository
from .module_factory import ModuleFactory
from .entities.entity import EntityType
from tqdm import tqdm


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

    def compute_test_static_metrics(self, entities, build_tc_features):
        for entity in entities:
            if entity.type == EntityType.TEST:
                for name, value in entity.metrics.items():
                    if name in DatasetFactory.COM_static_feature_set:
                        build_tc_features.setdefault(entity.id, {})
                        build_tc_features[entity.id][f"COM_{name}"] = value
        return build_tc_features

    def create_dataset(self, builds):
        builds.sort(key=lambda e: e.id)
        dataset = []
        for build in tqdm(builds, desc="Creating dataset"):
            try:
                self.git_repository.repo.git.checkout(build.commit_hash)
            except Exception as e:
                continue

            build_tc_features = {}
            code_analyzer = ModuleFactory.get_code_analyzer(self.level)
            with code_analyzer(
                self.project_path,
                self.test_path,
                self.output_path,
                self.language,
                self.level,
            ) as analyzer:
                entities = analyzer.get_entities()
                self.compute_test_static_metrics(entities, build_tc_features)

            for test_id, features in build_tc_features.items():
                features[DatasetFactory.BUILD] = build.id
                features[DatasetFactory.TEST] = test_id
                dataset.append(features)
        return dataset
