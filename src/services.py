
from src.dep_feature_extractor import *
from src.understand_database import *
import understand
import pandas as pd


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
	def compute_and_save_historical_data(args):
		print("Loading understand database ...")
		db = understand.open(args.db_path)

		output_dir = args.output_dir
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		extractor_type = DEPExtractor
		if args.level == "file":
			extractor_type = FileDEPExtractor
		elif args.level == "function":
			extractor_type = FunctionDEPExtractor

		understand_db_type = DataCollectionService.get_understand_db_type(args.language)
		understand_db = understand_db_type(db, args.level, args.project_path)
		extractor = extractor_type(understand_db, args.language)
		metadata = extractor.extract_metadata()
		metadata_df = pd.DataFrame(metadata)
		metadata_cols = metadata_df.columns.values.tolist()
		if 'UniqueName' in metadata_cols:
			metadata_cols.remove('UniqueName')
		metadata_df.to_csv(f"{output_dir}/metadata.csv", index=False, columns=metadata_cols)

		structural_graph = extractor.extract_structural_dependency_graph(metadata_df)
		miner_type = extractor.get_association_miner()
		repository = RepositoryMining(args.project_path, since=args.since, only_no_merge=True, only_in_branch=args.branch)
		miner = miner_type(repository, metadata_df, understand_db)
		association_map = miner.compute_association_map()
		logical_graph = extractor.extract_logical_dependency_graph(structural_graph, association_map)

		dep_df = pd.DataFrame({'entity_id': list(structural_graph.keys()), 'dependencies': list(structural_graph.values())})
		dep_df['weights'] = [logical_graph[ent_id] for ent_id in structural_graph.keys()]
		dep_df.to_csv(f"{output_dir}/dep_graph.csv", sep=';', index=False)
		print(f'All finished, results are saved in {output_dir}')

	@staticmethod
	def compute_and_save_release_data(args):
		commits_file = open(args.commits_file, 'r')
		commits = list(map(str.strip, commits_file.readlines()))
		repository = RepositoryMining(args.project_path, only_commits=commits)
		metadata_df = pd.read_csv(f'{args.histories_dir}/metadata.csv')

		understand_db_type = DataCollectionService.get_understand_db_type(args.language)
		understand_db = understand_db_type(None, args.level, args.project_path)

		miner_type = AssociationMiner
		if args.level == "file":
			miner_type = FileAssociationMiner
		elif args.level == "function":
			miner_type = FunctionAssociationMiner
		miner = miner_type(repository, metadata_df, understand_db)

		changed_sets = miner.compute_changed_sets()
		changed_entities = set.union(*changed_sets)
		dep_graph = pd.read_csv(f'{args.histories_dir}/dep_graph.csv', sep=';',
														converters={'dependencies': json.loads, 'weights': json.loads})
		changed_dep_graph = dep_graph[dep_graph.entity_id.isin(changed_entities)]
		changed_ids = []
		change_weights = []
		for r_index, row in changed_dep_graph.iterrows():
			changed_ids.append(row['entity_id'])
			change_weights.append(1)
			for i, dep in enumerate(row['dependencies']):
				changed_ids.append(dep)
				change_weights.append(row['weights'][i])

		output_dir = args.output_dir
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		release_changes = pd.DataFrame({'entity_id': changed_ids, 'weight': change_weights})
		release_changes.to_csv(f"{args.output_dir}/release_changes.csv", sep=';', index=False)
		print(f'All finished, results are saved in {output_dir}')
