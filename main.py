from src.structural_feature_extractor import *
import argparse
import understand
import pandas as pd
import os


def extract_and_save_structural_features(db, args):
	extractor_type = StructuralFeatureExtractor
	if args.level == "file":
		extractor_type = FileStructuralFeatureExtractor
	elif args.level == "function":
		extractor_type = FunctionStructuralFeatureExtractor

	extractor = extractor_type(db, args.language)
	metadata = extractor.extract_metadata()
	dep_graph = extractor.extract_dependency_graph()

	output_dir = args.output_dir
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	dep_df = pd.DataFrame({'entity_id': list(dep_graph.keys()), 'dependencies': list(dep_graph.values())})
	dep_df.to_csv(f"{output_dir}/dep_graph.csv", sep=';', index=False)
	metadata_df = pd.DataFrame(metadata)
	metadata_df.to_csv(f"{output_dir}/metadata.csv", index=False)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--db-path', help="Understand's database path with the .udb format.", required=True)
	parser.add_argument('-l', '--level', help="Specifies the granularity of feature extraction.",
											choices=['function', 'file'], required=True)
	parser.add_argument('-o', '--output-dir', help="Specifies the directory to save resulting datasets.", required=True)
	parser.add_argument('--language', help="Project's main language", default="c", choices=["c"])
	args = parser.parse_args()

	db = understand.open(args.db_path)
	extract_and_save_structural_features(db, args)


if __name__ == "__main__":
	main()
