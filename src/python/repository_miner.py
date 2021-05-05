from .entities.entity_change import EntityChange, Contributor
from .entities.entity import Language, Entity
from .entities.dep_graph import DepGraph
from typing import List
from pydriller.domain.commit import ModificationType
from pydriller import RepositoryMining, GitRepository
import pandas as pd
from tqdm import tqdm
import numpy as np
from .id_mapper import IdMapper
from apyori import apriori
from git import Git


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
        self.merged_commits = None
        self.merged_commit_list_d = None
        self.commit_change_list_d = None

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

    def compute_entity_change_history(self) -> List[EntityChange]:
        from .services import DataCollectionService

        DataCollectionService.checkout_default_branch(self.config.project_path)
        change_history = []
        repository = RepositoryMining(str(self.config.project_path))
        commits = list(repository.traverse_commits())
        merge_map = {}
        self.merged_commits = set()
        self.merged_commit_list_d = {}
        for commit in tqdm(commits, desc="Mining entity change history"):
            change_history.extend(self.compute_changed_entities(commit))
            if commit.merge:
                self.merged_commit_list_d.setdefault(commit.hash, [])
                for merged_commit in self.get_merge_commits(commit):
                    merge_map[merged_commit.hash] = commit.hash
                    self.merged_commits.add(merged_commit.hash)
                    self.merged_commit_list_d[commit.hash].append(merged_commit.hash)

        for change in tqdm(change_history, desc="Detecting merge commits"):
            if change.commit_hash in merge_map:
                change.merge_commit = merge_map[change.commit_hash]

        change_history_df = pd.DataFrame.from_records(
            [ch.to_dict() for ch in change_history]
        )
        commit_change_lists = (
            change_history_df[[EntityChange.ID, EntityChange.COMMIT]]
            .groupby(EntityChange.COMMIT)[EntityChange.ID]
            .apply(list)
            .reset_index(name="changes")
        )
        self.commit_change_list_d = dict(
            zip(
                commit_change_lists[EntityChange.COMMIT].values.tolist(),
                commit_change_lists["changes"].values.tolist(),
            )
        )

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

    def get_analysis_path(self, commit_hash):
        return self.config.output_path / "analysis" / commit_hash

    def analyze_commit_statically(self, commit_hash):
        from .module_factory import ModuleFactory

        analysis_path = self.get_analysis_path(commit_hash)
        metadata_path = analysis_path / "metadata.csv"
        if not metadata_path.exists():
            try:
                self.git_repository.repo.git.checkout(commit_hash)
            except:
                return pd.DataFrame()
            code_analyzer = ModuleFactory.get_code_analyzer(self.config.level)
            with code_analyzer(self.config, analysis_path) as analyzer:
                entities = analyzer.get_entities()
                entities_df = pd.DataFrame.from_records([e.to_dict() for e in entities])
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                entities_df.sort_values(by=[Entity.ID], ignore_index=True, inplace=True)
                entities_df.to_csv(metadata_path, index=False)
            return entities_df
        else:
            return pd.read_csv(metadata_path)

    def compute_dependency_weights(self, graph, co_changes):
        association_rules = apriori(
            co_changes,
            min_support=0.0001,
            min_confidence=0.002,
            max_length=2,
        )
        association_map = {rule.items: rule for rule in association_rules}

        for edge_start, dependencies in graph.graph.items():
            dep_weights = []
            for edge_end in dependencies:
                pair = frozenset({edge_start, edge_end})
                if pair in association_map:
                    rule = association_map[pair]
                    forward_confidence = 0.0
                    backward_confidence = 0.0
                    lift = 0.0
                    for stats in rule.ordered_statistics:
                        lift = stats.lift
                        if stats.items_base == frozenset({edge_end}):
                            forward_confidence = stats.confidence
                        elif stats.items_base == frozenset({edge_start}):
                            backward_confidence = stats.confidence
                    dep_weights.append(
                        [rule.support, forward_confidence, backward_confidence, lift]
                    )
                else:
                    dep_weights.append(0)
            graph.add_weights(edge_start, dep_weights)
        return graph

    def compute_co_changes(self, commit_hash, ent_ids_set):
        from .services import DataCollectionService

        if (
            (self.merged_commits is None)
            or (self.merged_commit_list_d is None)
            or (self.commit_change_list_d is None)
        ):
            self.compute_and_save_entity_change_history()
        build_commit = self.git_repository.get_commit(commit_hash)
        commit_date = build_commit.author_date
        DataCollectionService.checkout_default_branch(self.config.project_path)
        repository = RepositoryMining(str(self.config.project_path), to=commit_date)
        co_changes = []
        for commit in repository.traverse_commits():
            if commit.hash in self.merged_commits:
                continue
            changes = self.get_changed_entities(commit)
            entity_changes = changes.intersection(ent_ids_set)
            if len(entity_changes) > 0:
                co_changes.append(entity_changes)
        return co_changes

    def analyze_commit_dependency(self, commit_hash, test_ids, src_ids):
        from .module_factory import ModuleFactory

        test_ids.sort()
        src_ids.sort()
        analysis_path = self.get_analysis_path(commit_hash)
        dep_path = analysis_path / "dep.csv"
        tar_path = analysis_path / "tar.csv"
        if (not dep_path.exists()) or (not tar_path.exists()):
            try:
                self.git_repository.repo.git.checkout(commit_hash)
            except:
                return pd.DataFrame(), pd.DataFrame()
            co_changes = self.compute_co_changes(
                commit_hash, set(test_ids) | set(src_ids)
            )
            code_analyzer = ModuleFactory.get_code_analyzer(self.config.level)
            with code_analyzer(self.config, analysis_path) as analyzer:
                dep_graph = analyzer.compute_dependency_graph(src_ids, src_ids)
                tar_graph = analyzer.compute_dependency_graph(test_ids, src_ids)
                dep_graph = self.compute_dependency_weights(dep_graph, co_changes)
                tar_graph = self.compute_dependency_weights(tar_graph, co_changes)
                dep_graph.save_graph(dep_path, self.config.unique_separator)
                tar_graph.save_graph(
                    tar_path,
                    self.config.unique_separator,
                )
            return dep_graph, tar_graph
        else:
            dep_graph = DepGraph()
            tar_graph = DepGraph()
            dep_graph.load_graph(dep_path, self.config.unique_separator)
            tar_graph.load_graph(tar_path, self.config.unique_separator)
            return dep_graph, tar_graph

    def get_changed_entities(self, commit):
        try:
            changes = set()
            commit_hashes = [commit.hash]
            if commit.merge:
                commit_hashes = self.merged_commit_list_d[commit.hash]
            for change_hash in commit_hashes:
                changes.update(self.commit_change_list_d[change_hash])
            return changes
        except:
            return set()

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
                None,
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
                        changed_method_id,
                        added,
                        deleted,
                        contributor_id,
                        commit.hash,
                        commit.author_date,
                        None,
                    )
                    changed_entities.append(entity_change)
        return changed_entities
