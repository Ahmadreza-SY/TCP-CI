from pydriller import RepositoryMining
from pydriller.domain.commit import ModificationType
from datetime import datetime
from tqdm import tqdm
from apyori import apriori


class AssociationMiner:
	def __init__(self, project_path, metadata, since, branch):
		self.repository = RepositoryMining(project_path, since=since, only_no_merge=True, only_in_branch=branch)
		self.metadata = metadata
		self.min_support = 0.0001
		self.min_confidence = 0.002

	def compute_changed_set(self, commit):
		pass

	def compute_changed_sets(self):
		changed_sets = []
		global_changed_set = set()
		commit_count = 0
		for commit in tqdm(self.repository.traverse_commits(), desc="Traversing commits"):
			commit_count += 1
			changed_set = self.compute_changed_set(commit)
			if len(changed_set) > 0:
				changed_sets.append(changed_set)
				global_changed_set = global_changed_set.union(changed_set)
		print(f"Found {len(changed_sets)} change transactions with the total of {len(global_changed_set)} entities among {commit_count} commits")
		return changed_sets

	def compute_association_map(self):
		changed_sets = self.compute_changed_sets()
		print('Mining association rules ...')
		association_rules = apriori(changed_sets, min_support=self.min_support, min_confidence=self.min_confidence, max_length=2)
		return {rule.items: rule for rule in association_rules}


class FileAssociationMiner(AssociationMiner):
	def compute_changed_set(self, commit):
		changed_set = set()
		for m in commit.modifications:
			changed_file_path = None
			if m.change_type in [ModificationType.ADD, ModificationType.COPY, ModificationType.RENAME]:
				changed_file_path = m.new_path
			elif m.change_type in [ModificationType.DELETE, ModificationType.MODIFY]:
				changed_file_path = m.old_path
			changed_file_id = self.metadata[self.metadata.FilePath == changed_file_path] if changed_file_path is not None else None
			if changed_file_id is not None and len(changed_file_id) > 0:
				changed_set.add(changed_file_id.Id.values[0])
		return changed_set


class FunctionAssociationMiner(AssociationMiner):
	def compute_changed_set(self, commit):
		changed_set = set()
		for mod in commit.modifications:
			for method in mod.changed_methods:
				changed_method_meta = self.metadata[self.metadata.FullName == method.name]
				if len(changed_method_meta) == 1:
					changed_set.add(changed_method_meta.Id.values[0])
					continue
				elif len(changed_method_meta) > 1:
					changed_method_meta = changed_method_meta[changed_method_meta.FilePath.str.contains(method.filename)]
					if len(changed_method_meta) == 1:
						changed_set.add(changed_method_meta.Id.values[0])
						continue
		return changed_set
