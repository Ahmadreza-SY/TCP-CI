from src.python.dep_feature_extractor import *
from src.python.understand_database import *
import understand
import pandas as pd
import time
import subprocess
import shlex
import os
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
	def run_und_command(command, args):
		pbar = None
		full_project_path = os.path.abspath(args.project_path)
		process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
		while True:
			output = process.stdout.readline().decode('utf-8').strip()
			if output == '' and process.poll() is not None:
				break
			if output:
				if "Files added:" in output:
					pbar_total = int(output.split(' ')[-1])
					if args.language == "java":
						pbar_total *= 2
					pbar = tqdm(total=pbar_total, file=sys.stdout, desc="Analyzing files ...")
				elif "File:" not in output and "Warning:" not in output and pbar is not None:
					if "RELATIVE:/" in output or full_project_path in output:
						pbar.update(1)
		pbar.close()
		rc = process.poll()
		return rc

	@staticmethod
	def create_understand_database(args):
		project_name = args.project_path.split('/')[-1]
		und_db_path = f'{args.output_dir}/{project_name}.udb'
		if not os.path.isfile(und_db_path):
			start = time.time()
			language = 'c++' if args.language == 'c' else args.language
			print('Running understand analysis')
			und_command = f'und -verbose -db {und_db_path} create -languages {language} add {args.project_path} analyze'
			DataCollectionService.run_und_command(und_command, args)
			print(f'Created understand db at {und_db_path}, took {"{0:.2f}".format(time.time() - start)} seconds.')
		else:
			print(f'Understand db already exists at {und_db_path}, skipping understand analysis.')
		return und_db_path

	@staticmethod
	def save_dependency_graph(graph, weights, file_path, dependency_col_name):
		graph_df = pd.DataFrame({'entity_id': list(graph.keys()), dependency_col_name: list(graph.values())})
		graph_df['weights'] = [weights[ent_id] for ent_id in graph.keys()]
		graph_df.to_csv(file_path, sep=';', index=False)

	@staticmethod
	def compute_and_save_historical_data(args):
		print("Loading understand database ...")
		db = understand.open(args.db_path)

		output_dir = args.output_dir

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
		metadata_df.to_csv(f"{output_dir}/metadata.csv", index=False, columns=metadata_cols)
		test_count = len(metadata_df[metadata_df[DEPExtractor.ENTITY_TYPE_FIELD] == EntityType.TEST.name])
		print(f'Found a total of {len(metadata_df)} entities including {test_count} among test code and {len(metadata_df) - test_count} among the main source code.')

		dep_graph = extractor.create_static_dep_graph(metadata_df)
		tar_graph = extractor.create_static_tar_graph(metadata_df)
		miner_type = extractor.get_association_miner()
		repository = RepositoryMining(args.project_path, since=args.since, only_no_merge=True, only_in_branch=args.branch)
		miner = miner_type(repository, metadata_df, understand_db)
		association_map = miner.compute_association_map()
		dep_weights = extractor.extract_historical_dependency_weights(dep_graph, association_map, "Extracting DEP weights")
		tar_weights = extractor.extract_historical_dependency_weights(tar_graph, association_map, "Extracting TAR weights")

		DataCollectionService.save_dependency_graph(dep_graph, dep_weights, f"{output_dir}/dep.csv", 'dependencies')

		tar_reversed_graph = {}
		tar_reversed_weights = {}
		for test_id, src_ids in tar_graph.items():
			for i, src_id in enumerate(src_ids):
				tar_reversed_graph.setdefault(src_id, [])
				tar_reversed_weights.setdefault(src_id, [])
				tar_reversed_graph[src_id].append(test_id)
				tar_reversed_weights[src_id].append(tar_weights[test_id][i])
		DataCollectionService.save_dependency_graph(tar_reversed_graph, tar_reversed_weights,
																								f"{output_dir}/tar.csv", 'targeted_by_tests')
		print(f'All finished, results are saved in {output_dir}')

	@staticmethod
	def compute_and_save_release_data(args):
		commits_file = open(args.commits_file, 'r')
		commits = list(map(str.strip, commits_file.readlines()))
		repository = RepositoryMining(args.project_path, only_commits=commits)
		metadata_df = pd.read_csv(f'{args.histories_dir}/metadata.csv')

		understand_db_type = DataCollectionService.get_understand_db_type(args.language)
		understand_db = understand_db_type(None, args.level, args.project_path, args.test_path)

		miner_type = AssociationMiner
		if args.level == "file":
			miner_type = FileAssociationMiner
		elif args.level == "function":
			miner_type = FunctionAssociationMiner
		miner = miner_type(repository, metadata_df, understand_db)

		changed_sets = miner.compute_changed_sets()
		changed_entities = set.union(*changed_sets) if len(changed_sets) > 0 else set()
		dep_graph = pd.read_csv(f'{args.histories_dir}/dep.csv', sep=';',
														converters={'dependencies': json.loads, 'weights': json.loads})
		tar_graph = pd.read_csv(f'{args.histories_dir}/tar.csv', sep=';',
														converters={'targeted_by_tests': json.loads, 'weights': json.loads})
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

		output_dir = args.output_dir
		release_changes = pd.DataFrame({'entity_id': changed_ids, 'weight': change_weights,
																		'targeted_by_tests': targeted_by_tests, 'targeted_by_tests_weights': targeted_by_tests_weights})
		release_changes.to_csv(f"{args.output_dir}/release_changes.csv", sep=';', index=False)
		print(f'All finished, results are saved in {output_dir}')