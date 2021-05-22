from .entities.entity_change import EntityChange, Contributor
from .entities.entity import Language, Entity
from .entities.dep_graph import DepGraph
from .commit_classifier.commit_classifier import CommitClassifier, CommitType
from typing import List
from pydriller.domain.commit import ModificationType, DMMProperty
from pydriller import RepositoryMining, GitRepository
import pandas as pd
from tqdm import tqdm
import numpy as np
from .id_mapper import IdMapper
from apyori import apriori
from git import Git
import more_itertools as mit
from itertools import combinations
import sys
import re
from .timer import tik, tok


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
        self.commit_change_list_d = None
        self.commit_clf = CommitClassifier()

    def checkout_default_branch(self):
        g = Git(self.config.project_path)
        remote = g.execute("git remote show".split())
        if remote == "":
            print("Git repository has no remote! Please set a remote.")
            sys.exit()
        result = g.execute(f"git remote show {remote}".split())
        default_branch = re.search("HEAD branch: (.+)", result).groups()[0]
        git_repository = GitRepository(self.config.project_path)
        git_repository.repo.git.checkout(default_branch)

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

    def get_all_commits(self):
        self.checkout_default_branch()
        repository = RepositoryMining(str(self.config.project_path))
        return list(repository.traverse_commits())

    def compute_entity_change_history(self) -> List[EntityChange]:
        change_history = []
        commits = self.get_all_commits()
        for commit in tqdm(commits, desc="Mining entity change history"):
            tik("Compute Commit Modifications")
            modifications = commit.modifications
            if commit.merge:
                diff_index = commit._c_object.parents[0].diff(
                    commit._c_object, create_patch=True
                )
                modifications = commit._parse_diff(diff_index)
            tok("Compute Commit Modifications")
            tik("Compute Changed Ents")
            change_history.extend(self.compute_changed_entities(commit, modifications))
            tok("Compute Changed Ents")

        tik("Create Commit Change List")
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
        tok("Create Commit Change List")

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

    def analyze_commit_statically(self, build):
        from .module_factory import ModuleFactory

        analysis_path = self.get_analysis_path(build.commit_hash)
        metadata_path = analysis_path / "metadata.csv"
        if not metadata_path.exists():
            try:
                self.git_repository.repo.git.checkout(build.commit_hash)
            except:
                return pd.DataFrame()
            tik("Static Analysis", build.id)
            code_analyzer = ModuleFactory.get_code_analyzer(self.config.level)
            with code_analyzer(self.config, analysis_path) as analyzer:
                entities = analyzer.get_entities()
                if len(entities) == 0:
                    entities_df = pd.DataFrame()
                else:
                    entities_df = pd.DataFrame.from_records(
                        [e.to_dict() for e in entities]
                    )
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    entities_df.sort_values(
                        by=[Entity.ID], ignore_index=True, inplace=True
                    )
                    entities_df.to_csv(metadata_path, index=False)
            tok("Static Analysis", build.id)
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
        if self.commit_change_list_d is None:
            self.compute_and_save_entity_change_history()
        build_commit = self.git_repository.get_commit(commit_hash)
        commit_date = build_commit.committer_date
        self.checkout_default_branch()
        repository = RepositoryMining(str(self.config.project_path), to=commit_date)
        co_changes = []
        for commit in repository.traverse_commits():
            if commit.merge:
                continue
            changes = self.get_changed_entities(commit)
            entity_changes = changes.intersection(ent_ids_set)
            if len(entity_changes) > 0:
                co_changes.append(entity_changes)
        return co_changes

    def analyze_commit_dependency(self, build, test_ids, src_ids):
        from .module_factory import ModuleFactory

        test_ids.sort()
        src_ids.sort()
        analysis_path = self.get_analysis_path(build.commit_hash)
        dep_path = analysis_path / "dep.csv"
        tar_path = analysis_path / "tar.csv"
        if (not dep_path.exists()) or (not tar_path.exists()):
            try:
                self.git_repository.repo.git.checkout(build.commit_hash)
            except:
                return pd.DataFrame(), pd.DataFrame()
            tik("Dependency Analysis", build.id)
            co_changes = self.compute_co_changes(
                build.commit_hash, set(test_ids) | set(src_ids)
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
            tok("Dependency Analysis", build.id)
            return dep_graph, tar_graph
        else:
            dep_graph = DepGraph()
            tar_graph = DepGraph()
            dep_graph.load_graph(dep_path, self.config.unique_separator)
            tar_graph.load_graph(tar_path, self.config.unique_separator)
            return dep_graph, tar_graph

    def get_changed_entities(self, commit):
        return set(self.commit_change_list_d[commit.hash])

    def compute_changed_entities(self, commit, modifications) -> List[EntityChange]:
        pass

    def compute_function_diff(self, func, diff_parsed):
        added = []
        for diff in diff_parsed["added"]:
            if func.start_line <= diff[0] <= func.end_line:
                added.append(diff[0])
        deleted = []
        for diff in diff_parsed["deleted"]:
            if func.start_line <= diff[0] <= func.end_line:
                deleted.append(diff[0])
        return added, deleted

    def compute_scattering(self, changed_lines):
        changed_lines.sort()
        chunks = [list(group) for group in mit.consecutive_groups(changed_lines)]
        if len(chunks) <= 1:
            return 0.0
        combs = list(combinations(chunks, 2))
        distance_sum = sum(map(lambda pair: abs(pair[0][0] - pair[1][0]), combs))
        return (len(chunks) / len(combs)) * distance_sum

    def get_commit_class(self, commit_msg):
        commit_cls = CommitType.NONE_BUG
        if commit_msg and commit_msg != "":
            commit_cls = self.commit_clf.classify_commit(commit_msg)
        return commit_cls


class FileRepositoryMiner(RepositoryMiner):
    def compute_dmm(self, modification, dmm_prop):
        lr_changes = 0
        hr_changes = 0
        diff_parsed = modification.diff_parsed
        for method in modification.changed_methods:
            added_changes, deleted_changes = self.compute_function_diff(
                method, diff_parsed
            )
            changes = len(added_changes) + len(deleted_changes)
            if method.is_low_risk(dmm_prop):
                lr_changes += changes
            else:
                hr_changes += changes
        return (lr_changes, hr_changes)

    def compute_changed_entities(self, commit, modifications):
        changed_entities = []
        contributor_id = self.get_contributor_id(commit)
        for mod in modifications:
            tik("Get Entity Id")
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
            tok("Get Entity Id")

            tik("Change Scattering")
            diff = mod.diff_parsed
            added_lines = list(map(lambda p: p[0], diff["added"]))
            added_scattering = self.compute_scattering(added_lines)
            deleted_lines = list(map(lambda p: p[0], diff["deleted"]))
            deleted_scattering = self.compute_scattering(deleted_lines)
            tok("Change Scattering")
            tik("DMM")
            dmm_size = self.compute_dmm(mod, DMMProperty.UNIT_SIZE)
            dmm_complexity = self.compute_dmm(mod, DMMProperty.UNIT_COMPLEXITY)
            dmm_interfacing = self.compute_dmm(mod, DMMProperty.UNIT_INTERFACING)
            tok("DMM")
            tik("Commit Classification")
            commit_class = self.get_commit_class(commit.msg).value
            tok("Commit Classification")
            tik("Create EntityChange")
            changed_entity = EntityChange(
                changed_file_id,
                mod.added,
                mod.removed,
                (added_scattering, deleted_scattering),
                dmm_size,
                dmm_complexity,
                dmm_interfacing,
                contributor_id,
                commit_class,
                commit.hash,
                commit.committer_date,
                commit.merge,
            )
            changed_entities.append(changed_entity)
            tok("Create EntityChange")
        return changed_entities


class FunctionRepositoryMiner(RepositoryMiner):
    def compute_dmm(self, changed_method, dmm_prop):
        if changed_method.is_low_risk(dmm_prop):
            return (1, 0)
        else:
            return (0, 1)

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

    def compute_changed_entities(self, commit, modifications):
        changed_entities = []
        contributor_id = self.get_contributor_id(commit)
        for mod in modifications:
            diff_parsed = mod.diff_parsed
            for method in mod.changed_methods:
                method_unique_name = self.get_pydriller_function_unique_name(method)
                if method_unique_name is not None:
                    changed_method_id = self.id_mapper.get_entity_id(method_unique_name)
                    added, deleted = self.compute_function_diff(method, diff_parsed)
                    added_scattering = self.compute_scattering(added)
                    deleted_scattering = self.compute_scattering(deleted)
                    entity_change = EntityChange(
                        changed_method_id,
                        len(added),
                        len(deleted),
                        (added_scattering, deleted_scattering),
                        self.compute_dmm(mod, DMMProperty.UNIT_SIZE),
                        self.compute_dmm(mod, DMMProperty.UNIT_COMPLEXITY),
                        self.compute_dmm(mod, DMMProperty.UNIT_INTERFACING),
                        contributor_id,
                        self.get_commit_class(commit.msg).value,
                        commit.hash,
                        commit.committer_date,
                        commit.merge,
                    )
                    changed_entities.append(entity_change)
        return changed_entities
