from .entities.entity_change import EntityChange, Contributor
from .entities.entity import Language, Entity, Function
from typing import List
from pydriller.domain.commit import ModificationType
from pydriller import RepositoryMining, GitRepository
import pandas as pd
from tqdm import tqdm
import numpy as np
from .id_mapper import IdMapper


class RepositoryMiner:
    def __init__(self, config):
        self.config = config
        self.contributors_path = config.output_path / "contributors.csv"
        if self.contributors_path.exists():
            contributors_df = pd.read_csv(self.contributors_path)
            self.contributors = dict(
                zip(
                    contributors_df[Contributor.KEY].values.tolist(),
                    map(Contributor.from_dict, contributors_df.to_dict("records")),
                )
            )
            self.max_id = np.max(contributors_df[Contributor.ID].values)
        else:
            self.contributors = {}
            self.max_id = 0
        self.id_mapper = IdMapper(config.output_path)
        self.git_repository = GitRepository(str(self.config.project_path))

    def get_contributor_id(self, commit):
        contributor = commit.author
        key = contributor.email if contributor.email is not None else contributor.name
        if key in self.contributors:
            return self.contributors[key].id

        self.max_id += 1
        new_contributor = Contributor(
            self.max_id, key, contributor.name, contributor.email
        )
        self.contributors[key] = new_contributor
        return new_contributor.id

    def save_contributors(self):
        contributors_df = pd.DataFrame.from_records(
            [c.to_dict() for c in self.contributors.values()]
        )
        contributors_df.to_csv(self.contributors_path, index=False)

    def compute_entity_change_history(self) -> List[EntityChange]:
        change_history = []
        repository = RepositoryMining(str(self.config.project_path))
        commits = list(repository.traverse_commits())
        for commit in tqdm(commits, desc="Mining entity change history"):
            change_history.extend(self.compute_changed_entities(commit))
        self.save_contributors()
        self.id_mapper.save_id_map()
        return change_history

    def compute_and_save_entity_change_history(self) -> List[EntityChange]:
        change_history = self.compute_entity_change_history()
        change_history_df = pd.DataFrame.from_records(
            [ch.to_dict() for ch in change_history]
        )
        change_history_df.to_csv(
            self.config.output_path / "entity_change_history.csv", index=False
        )
        return change_history_df

    def analyze_commit(self, commit_hash):
        from .module_factory import ModuleFactory

        metadata_path = (
            self.config.output_path / "metadata" / commit_hash / "metadata.csv"
        )
        if not metadata_path.exists():
            try:
                self.git_repository.repo.git.checkout(commit_hash)
            except:
                return pd.DataFrame()
            code_analyzer = ModuleFactory.get_code_analyzer(self.config.level)
            with code_analyzer(self.config) as analyzer:
                entities = analyzer.get_entities()
                entities_df = pd.DataFrame.from_records([e.to_dict() for e in entities])
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                entities_df.to_csv(metadata_path, index=False)
            return entities_df
        else:
            return pd.read_csv(metadata_path)

    def compute_changed_entities(self, commit) -> List[EntityChange]:
        pass


class FileRepositoryMiner(RepositoryMiner):
    def compute_changed_entities(self, commit):
        changed_entities = []
        contributor_id = self.get_contributor_id(commit)
        for mod in commit.modifications:
            changed_file_id = None
            if mod.change_type in [
                ModificationType.ADD,
                ModificationType.COPY,
            ]:
                changed_file_id = self.id_mapper.get_entity_id(mod.new_path)
            elif mod.change_type in [
                ModificationType.DELETE,
                ModificationType.MODIFY,
            ]:
                changed_file_id = self.id_mapper.get_entity_id(mod.old_path)
            elif mod.change_type == ModificationType.RENAME:
                changed_file_id = self.id_mapper.merge_entity_ids(
                    (mod.old_path, mod.new_path)
                )
            else:
                changed_file_id = self.id_mapper.get_entity_id(mod.new_path)
            changed_entity = EntityChange(
                changed_file_id,
                mod.added,
                mod.removed,
                contributor_id,
                commit.hash,
                commit.author_date,
            )
            changed_entities.append(changed_entity)
        return changed_entities


class FunctionRepositoryMiner(RepositoryMiner):
    def get_pydriller_function_unique_name(self, pydriller_function):
        if self.config.language == Language.JAVA:
            return pydriller_function.long_name.replace(" ", "")
        elif self.config.language == Language.C:
            function_name = pydriller_function.name
            if function_name == "TEST":
                function_name = pydriller_function.long_name
                test_names = (
                    function_name[function_name.find("(") + 1 : function_name.find(")")]
                    .replace(" ", "")
                    .split(",")
                )
                function_name = f"{test_names[0]}_{test_names[1]}_Test::TestBody"
            return function_name
        return None

    def compute_function_diff(self, func, diff_parsed):
        added = 0
        for diff in diff_parsed["added"]:
            if func.start_line <= diff[0] <= func.end_line:
                added += 1
        deleted = 0
        for diff in diff_parsed["deleted"]:
            if func.start_line <= diff[0] <= func.end_line:
                deleted += 1
        return added, deleted

    def compute_changed_entities(self, commit):
        changed_entities = []
        contributor_id = self.get_contributor_id(commit)
        for mod in commit.modifications:
            diff_parsed = mod.diff_parsed
            for method in mod.changed_methods:
                method_unique_name = self.get_pydriller_function_unique_name(method)
                if method_unique_name is not None:
                    changed_method_id = self.id_mapper.get_entity_id(method_unique_name)
                    added, deleted = self.compute_function_diff(method, diff_parsed)
                    entity_change = EntityChange(
                        changed_method_id, added, deleted, contributor_id, commit.hash
                    )
                    changed_entities.append(entity_change)
        return changed_entities
