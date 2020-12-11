from pydriller import RepositoryMining
from tqdm import tqdm


class CommitMiner:
	def __init__(self, project_path):
		self.repository = RepositoryMining(project_path)
		self.last_contributor_id = 0
		self.contributors = {}
		self.commit_features = {
				'hash': [],
				'committer': [],
				'committer_date': [],
				'author': [],
				'author_date': [],
				'in_main_branch': [],
				'merge': [],
				'dmm_unit_complexity': [],
				'dmm_unit_interfacing': [],
				'dmm_unit_size': [],
				'additions': [],
				'deletions': [],
				'avg_complexity': [],
				'max_complexity': [],
				'parents': []
		}

	def get_or_create_contributor_id(self, contributor):
		key = contributor.email if contributor.email is not None else contributor.name
		if key in self.contributors:
			return self.contributors[key]['id']

		self.last_contributor_id += 1
		new_contributor = {'id': self.last_contributor_id, 'name': contributor.name, 'email': contributor.email}
		self.contributors[key] = new_contributor
		return new_contributor['id']

	def extract_commit_features(self, commit):
		self.commit_features['hash'].append(commit.hash)
		self.commit_features['committer'].append(self.get_or_create_contributor_id(commit.committer))
		self.commit_features['committer_date'].append(commit.committer_date.strftime("%Y-%m-%d %H:%M"))
		self.commit_features['author'].append(self.get_or_create_contributor_id(commit.author))
		self.commit_features['author_date'].append(commit.author_date.strftime("%Y-%m-%d %H:%M"))
		self.commit_features['in_main_branch'].append(commit.in_main_branch)
		self.commit_features['merge'].append(commit.merge)
		self.commit_features['dmm_unit_complexity'].append(commit.dmm_unit_complexity)
		self.commit_features['dmm_unit_interfacing'].append(commit.dmm_unit_interfacing)
		self.commit_features['dmm_unit_size'].append(commit.dmm_unit_size)
		self.commit_features['additions'].append(sum(list(map(lambda m: m.added, commit.modifications))))
		self.commit_features['deletions'].append(sum(list(map(lambda m: m.removed, commit.modifications))))
		mods_complexity = list(filter(lambda c: c is not None, map(lambda m: m.complexity, commit.modifications)))
		avg_complexity = 0.0 if len(mods_complexity) == 0 else sum(mods_complexity) / len(mods_complexity)
		self.commit_features['avg_complexity'].append(avg_complexity)
		self.commit_features['max_complexity'].append(0.0 if len(mods_complexity) == 0 else max(mods_complexity))
		self.commit_features['parents'].append(commit.parents)

	def mine_commits(self):
		commits = list(self.repository.traverse_commits())
		commit_features = []
		for commit in tqdm(commits, desc="Extracting commits features ..."):
			self.extract_commit_features(commit)
		return self.commit_features, self.contributors
