from src.dep_feature_extractor import *
import argparse
import understand
import pandas as pd
import os


def extract_and_save_dep_features(db, args):
	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	extractor_type = DEPExtractor
	if args.level == "file":
		extractor_type = FileDEPExtractor
	elif args.level == "function":
		extractor_type = FunctionDEPExtractor

	extractor = extractor_type(db, args.language)
	metadata = extractor.extract_metadata()
	metadata_df = pd.DataFrame(metadata)
	metadata_df.to_csv(f"{output_dir}/metadata.csv", index=False)

	structural_graph = extractor.extract_structural_dependency_graph(metadata_df)
	miner_type = extractor.get_association_miner()
	miner = miner_type(args.project_path, metadata_df, args.since, args.branch)
	association_map = miner.compute_association_map()
	logical_graph = extractor.extract_logical_dependency_graph(structural_graph, association_map)

	dep_df = pd.DataFrame({'entity_id': list(structural_graph.keys()), 'dependencies': list(structural_graph.values())})
	dep_df['weights'] = [logical_graph[ent_id] for ent_id in structural_graph.keys()]
	dep_df.to_csv(f"{output_dir}/dep_graph.csv", sep=';', index=False)
	print(f'All finished, results are saved in {output_dir}')


def valid_date(s):
	try:
		return datetime.strptime(s, "%Y-%m-%d")
	except ValueError:
		msg = "Not a valid date: '{0}'.".format(s)
		raise argparse.ArgumentTypeError(msg)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--db-path', help="Understand's database path with the .udb format.", required=True)
	parser.add_argument('-p', '--project-path', help="Project's source code git repository path.", required=True)
	parser.add_argument('-l', '--level', help="Specifies the granularity of feature extraction.",
											choices=['function', 'file'], required=True)
	parser.add_argument('-o', '--output-dir', help="Specifies the directory to save resulting datasets.", required=True)
	parser.add_argument('--language', help="Project's main language", default="c", choices=["c"])
	parser.add_argument('--branch', help="Git branch to analyze.", default="master")
	parser.add_argument('--since',
											help="Start date for commits to analyze - format YYYY-MM-DD. Not providing this arguments means to analyze all commits.",
											type=valid_date,
											default=None)
	args = parser.parse_args()

	print("Loading understand database ...")
	db = understand.open(args.db_path)
	extract_and_save_dep_features(db, args)


if __name__ == "__main__":
	main()
