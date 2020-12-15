from src.python.dep_feature_extractor import *
from src.python.exe_feature_extractor import *
from src.python.understand_database import *
from src.python.understand_runner import *
from src.python.commit_miner import *
from src.python.release_feature_extractor import *
import understand
import pandas as pd
import os
from os.path import isfile
import sys
import json


class DataCollectionService:

	@staticmethod
	def get_understand_db_type(language):
		understand_db_type = UnderstandDatabase
		if language == "c":
			understand_db_type = CUnderstandDatabase
		elif language == "java":
			understand_db_type = JavaUnderstandDatabase
		return understand_db_type

	@staticmethod
	def create_understand_database(args):
		return UnderstandRunner.create_understand_database(args)

	@staticmethod
	def fetch_and_save_execution_history(args):
		ExeFeatureExtractor.fetch_and_save_execution_history(args)

	@staticmethod
	def save_dependency_graph(graph, weights, file_path, dependency_col_name):
		graph_df = pd.DataFrame({'entity_id': list(graph.keys()), dependency_col_name: list(graph.values())})
		graph_df['weights'] = [weights[ent_id] for ent_id in graph.keys()]
		graph_df.to_csv(file_path, sep=';', index=False)

	@staticmethod
	def compute_and_save_dep_data(extractor, metadata_df, understand_db, args):
		output_dir = args.output_dir
		if isfile(f"{output_dir}/metadata.csv") and isfile(f"{output_dir}/dep.csv") and isfile(f"{output_dir}/tar.csv"):
			print(f'Dependency datasets already exist, skipping dependency analysis.')
			return

		dep_graph = extractor.create_static_dep_graph(metadata_df)
		tar_graph = extractor.create_static_tar_graph(metadata_df)
		miner_type = extractor.get_association_miner()
		repository = RepositoryMining(args.project_path, since=args.since, only_in_branch=args.branch)
		miner = miner_type(repository, metadata_df, understand_db)
		association_map = miner.compute_association_map()
		dep_weights = extractor.extract_historical_dependency_weights(dep_graph, association_map, "Extracting DEP weights")
		tar_weights = extractor.extract_historical_dependency_weights(tar_graph, association_map, "Extracting TAR weights")
		DataCollectionService.save_dependency_graph(dep_graph, dep_weights, f"{args.output_dir}/dep.csv", 'dependencies')

		tar_reversed_graph = {}
		tar_reversed_weights = {}
		for test_id, src_ids in tar_graph.items():
			for i, src_id in enumerate(src_ids):
				tar_reversed_graph.setdefault(src_id, [])
				tar_reversed_weights.setdefault(src_id, [])
				tar_reversed_graph[src_id].append(test_id)
				tar_reversed_weights[src_id].append(tar_weights[test_id][i])
		DataCollectionService.save_dependency_graph(tar_reversed_graph, tar_reversed_weights,
																								f"{args.output_dir}/tar.csv", 'targeted_by_tests')

	@staticmethod
	def compute_and_save_commit_data(args):
		output_dir = args.output_dir
		if isfile(f"{output_dir}/commits.csv") and isfile(f"{output_dir}/contributors.csv"):
			print(f'Commmit datasets already exist, skipping commit mining.')
			return
		commit_miner = CommitMiner(args.project_path)
		commit_features, contributors = commit_miner.mine_commits()
		pd.DataFrame(commit_features).to_csv(f'{args.output_dir}/commits.csv', index=False, sep=';')
		pd.DataFrame(list(contributors.values())).to_csv(f'{args.output_dir}/contributors.csv', index=False)

	@staticmethod
	def compute_and_save_historical_data(args):
		print("Loading understand database ...")
		db = understand.open(args.db_path)

		extractor_type = DEPExtractor
		if args.level == "file":
			extractor_type = FileDEPExtractor
		elif args.level == "function":
			extractor_type = FunctionDEPExtractor

		understand_db_type = DataCollectionService.get_understand_db_type(args.language)
		understand_db = understand_db_type(db, args.level, args.project_path, args.test_path)
		extractor = extractor_type(understand_db, args.language)
		metadata = extractor.extract_metadata()
		metadata_df = pd.DataFrame(metadata)
		metadata_cols = metadata_df.columns.values.tolist()
		metadata_df.to_csv(f"{args.output_dir}/metadata.csv", index=False, columns=metadata_cols)
		test_count = len(metadata_df[metadata_df[DEPExtractor.ENTITY_TYPE_FIELD] == EntityType.TEST.name])
		print(f'Found a total of {len(metadata_df)} entities including {test_count} among test code and {len(metadata_df) - test_count} among the main source code.')

		DataCollectionService.compute_and_save_dep_data(extractor, metadata_df, understand_db, args)
		DataCollectionService.compute_and_save_commit_data(args)

	@staticmethod
	def compute_and_save_release_data(args):
		output_dir = args.output_dir
		repository = RepositoryMining(args.project_path, from_commit=args.from_commit, to_commit=args.to_commit)
		metadata_df = pd.read_csv(f'{args.histories_dir}/metadata.csv')

		understand_db_type = DataCollectionService.get_understand_db_type(args.language)
		understand_db = understand_db_type(None, args.level, args.project_path, args.test_path)

		miner_type = AssociationMiner
		if args.level == "file":
			miner_type = FileAssociationMiner
		elif args.level == "function":
			miner_type = FunctionAssociationMiner
		miner = miner_type(repository, metadata_df, understand_db)

		print('Extracting release impacts ...')
		changed_sets = miner.compute_changed_sets()
		changed_entities = set.union(*changed_sets) if len(changed_sets) > 0 else set()
		dep_graph = pd.read_csv(f'{args.histories_dir}/dep.csv', sep=';',
														converters={'dependencies': json.loads, 'weights': json.loads})
		tar_graph = pd.read_csv(f'{args.histories_dir}/tar.csv', sep=';',
														converters={'targeted_by_tests': json.loads, 'weights': json.loads})
		release_impacts = ReleaseFeatureExtractor.extract_release_impacts(changed_entities, dep_graph, tar_graph)
		release_impacts.to_csv(f"{output_dir}/release_impacts.csv", sep=';', index=False)

		print('Extracting release changes ...')
		release_changes = ReleaseFeatureExtractor.extract_release_changes(
				metadata_df, args.project_path, args.from_commit, args.to_commit)
		pd.DataFrame(release_changes).to_csv(f"{output_dir}/release_changes.csv", index=False)

		print(f'All finished, results are saved in {output_dir}')
