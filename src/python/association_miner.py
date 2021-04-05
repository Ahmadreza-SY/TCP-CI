from pydriller.domain.commit import ModificationType
from tqdm import tqdm
from apyori import apriori
from .entities.entity import Language


class AssociationMiner:
    def __init__(self, repository, metadata, language):
        self.repository = repository
        self.metadata = metadata
        self.min_support = 0.0001
        self.min_confidence = 0.002
        self.language = language
        self.association_map = None

    def compute_changed_set(self, commit):
        pass

    def compute_changed_sets(self):
        print("Reading commits ...")
        commits = list(self.repository.traverse_commits())
        changed_sets = []
        global_changed_set = set()
        for commit in tqdm(commits, desc="Looking for associations among commits ..."):
            changed_set = self.compute_changed_set(commit)
            if len(changed_set) > 0:
                changed_sets.append(changed_set)
                global_changed_set = global_changed_set.union(changed_set)
        print(
            f"Found {len(changed_sets)} change transactions with the total of {len(global_changed_set)} entities among {len(commits)} commits"
        )
        return changed_sets

    def get_pydriller_function_unique_name(self, pydriller_function):
        if self.language == Language.JAVA:
            return pydriller_function.long_name.replace(" ", "")
        elif self.language == Language.C:
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

    def compute_association_map(self):
        if self.association_map is not None:
            return self.association_map
        changed_sets = self.compute_changed_sets()
        print("Mining association rules ...")
        association_rules = apriori(
            changed_sets,
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            max_length=2,
        )
        self.association_map = {rule.items: rule for rule in association_rules}
        return self.association_map

    def compute_dependency_weights(self, dep_graph):
        association_map = self.compute_association_map()
        for edge_start, dependencies in tqdm(
            dep_graph.graph.items(), desc="Assigning weights to dependency graph"
        ):
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
            dep_graph.add_weights(edge_start, dep_weights)
        return dep_graph


class FileAssociationMiner(AssociationMiner):
    def compute_changed_set(self, commit):
        changed_set = set()
        for m in commit.modifications:
            changed_file_path = None
            if m.change_type in [
                ModificationType.ADD,
                ModificationType.COPY,
                ModificationType.RENAME,
            ]:
                changed_file_path = m.new_path
            elif m.change_type in [ModificationType.DELETE, ModificationType.MODIFY]:
                changed_file_path = m.old_path
            changed_file_id = (
                self.metadata[self.metadata.FilePath == changed_file_path]
                if changed_file_path is not None
                else None
            )
            if changed_file_id is not None and len(changed_file_id) > 0:
                changed_set.add(changed_file_id.Id.values[0])
        return changed_set


class FunctionAssociationMiner(AssociationMiner):
    def compute_changed_set(self, commit):
        changed_set = set()
        for mod in commit.modifications:
            for method in mod.changed_methods:
                method_unique_name = self.get_pydriller_function_unique_name(method)
                changed_method_meta = self.metadata[
                    self.metadata.UniqueName == method_unique_name
                ]
                if len(changed_method_meta) > 0:
                    changed_set.add(changed_method_meta.Id.values[0])
        return changed_set
