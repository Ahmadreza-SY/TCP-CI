from src.dep_feature_extractor import *
from src.understand_database import *
import argparse
import understand
import pandas as pd
import os
import sys
import json


def history(args):
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

	understand_db_type = UnderstandDatabase
	if args.language == "c":
		understand_db_type = CUnderstandDatabase
	elif args.language == "java":
		understand_db_type = JavaUnderstandDatabase

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


def release(args):
	commits_file = open(args.commits_file, 'r')
	commits = list(map(str.strip, commits_file.readlines()))
	repository = RepositoryMining(args.project_path, only_commits=commits)
	metadata_df = pd.read_csv(f'{args.histories_dir}/metadata.csv')

	understand_db_type = UnderstandDatabase
	if args.language == "c":
		understand_db_type = CUnderstandDatabase
	elif args.language == "java":
		understand_db_type = JavaUnderstandDatabase
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

	release_changes = pd.DataFrame({'entity_id': changed_ids, 'weight': change_weights})
	release_changes.to_csv(f"{args.output_dir}/release_changes.csv", sep=';', index=False)


def valid_date(s):
	try:
		return datetime.strptime(s, "%Y-%m-%d")
	except ValueError:
		msg = "Not a valid date: '{0}'.".format(s)
		raise argparse.ArgumentTypeError(msg)


def add_common_arguments(parser):
	parser.add_argument('-p', '--project-path', help="Project's source code git repository path.", required=True)
	parser.add_argument('-l', '--level', help="Specifies the granularity of feature extraction.",
											choices=['function', 'file'], required=True)
	parser.add_argument('-o', '--output-dir', help="Specifies the directory to save resulting datasets.", default=".")
	parser.add_argument('--language', help="Project's main language", choices=["c", "java"], required=True)


def main():
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers()
	history_parser = subparsers.add_parser("history", help="The history command extracts and computes code dependencies.")
	release_parser = subparsers.add_parser(
			"release", help="The release command extracts changed entities and their dependencies based on the pre-computed history.")

	add_common_arguments(history_parser)
	history_parser.set_defaults(func=history)
	history_parser.add_argument('-d', '--db-path', help="Understand's database path with the .udb format.", required=True)
	history_parser.add_argument('--branch', help="Git branch to analyze.", default="master")
	history_parser.add_argument('--since', help="Start date for commits to analyze - format YYYY-MM-DD. Not providing this arguments means to analyze all commits.",
															type=valid_date,
															default=None)

	add_common_arguments(release_parser)
	release_parser.set_defaults(func=release)
	release_parser.add_argument('-c', '--commits-file',
															help="Path to a text file including commit hashes of a release in each line.", required=True)
	release_parser.add_argument('-hist', '--histories-dir', help="Path to outputs of the history command.", required=True)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
