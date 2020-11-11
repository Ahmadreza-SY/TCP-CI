from src.services import *
import argparse
import os

def history(args):
	args.db_path = DataCollectionService.create_understand_database(args)
	DataCollectionService.compute_and_save_historical_data(args)

def release(args):
	DataCollectionService.compute_and_save_release_data(args)

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
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	args.func(args)


if __name__ == "__main__":
	main()
