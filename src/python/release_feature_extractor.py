import pandas as pd
from pydriller.metrics.process.change_set import ChangeSet
from pydriller.metrics.process.code_churn import CodeChurn
from pydriller.metrics.process.commits_count import CommitsCount
from pydriller.metrics.process.contributors_count import ContributorsCount
from pydriller.metrics.process.contributors_experience import ContributorsExperience
from pydriller.metrics.process.hunks_count import HunksCount
from pydriller.metrics.process.lines_count import LinesCount


class ReleaseFeatureExtractor:

	@staticmethod
	def extract_release_impacts(changed_entities, dep_graph, tar_graph):
		tar_graph_dict = dict(zip(tar_graph.entity_id.values, zip(tar_graph.targeted_by_tests.values, tar_graph.weights.values)))
		changed_dep_graph = dep_graph[dep_graph.entity_id.isin(changed_entities)]
		changed_ids = []
		change_weights = []
		targeted_by_tests = []
		targeted_by_tests_weights = []

		def update_targeted_by_tests(entity_id):
			if entity_id in tar_graph_dict:
				targeted_by_tests.append(tar_graph_dict[entity_id][0])
				targeted_by_tests_weights.append(tar_graph_dict[entity_id][1])
			else:
				targeted_by_tests.append([])
				targeted_by_tests_weights.append([])

		for r_index, row in changed_dep_graph.iterrows():
			entity_id = row['entity_id']
			changed_ids.append(entity_id)
			change_weights.append(1)
			update_targeted_by_tests(entity_id)
			for i, dep in enumerate(row['dependencies']):
				changed_ids.append(dep)
				change_weights.append(row['weights'][i])
				update_targeted_by_tests(dep)

		return pd.DataFrame({'entity_id': changed_ids, 'weight': change_weights, 'targeted_by_tests': targeted_by_tests,
												 'targeted_by_tests_weights': targeted_by_tests_weights})

	@staticmethod
	def extract_release_changes(metdata_df, project_path, from_commit, to_commit):
		file_path_to_id = dict(zip(metdata_df.FilePath, metdata_df.Id))
		release_changes = {}
		process_metrics = [
				(CodeChurn, 'code_churn', ['count', 'max', 'avg']),
				(CommitsCount, 'commits', ['count']),
				(ContributorsCount, 'contributors', ['count', 'count_minor']),
				(ContributorsExperience, 'contributors_experience', ['count']),
				(HunksCount, 'hunks', ['count']),
				(LinesCount, 'lines', ['count', 'count_added', 'max_added',
															 'avg_added', 'count_removed', 'max_removed', 'avg_removed'])
		]

		for pm in process_metrics:
			metric = pm[0](path_to_repo=project_path, from_commit=from_commit, to_commit=to_commit)
			for metric_method in pm[2]:
				metric_values = getattr(metric, metric_method)()
				for file_path, value in metric_values.items():
					if file_path not in file_path_to_id:
						continue
					if file_path not in release_changes:
						release_changes[file_path] = {}
					change = release_changes[file_path]
					if 'entity_id' not in change:
						change['entity_id'] = file_path_to_id[file_path]
					change[f'{pm[1]}_{metric_method}'] = value

		metric = ChangeSet(path_to_repo=project_path, from_commit=from_commit, to_commit=to_commit)
		for file_path, change in release_changes.items():
			change['change_set_max'] = metric.max()
			change['change_set_avg'] = metric.avg()

		return release_changes.values()
