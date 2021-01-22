from src.python.services import *
import argparse
import os
import sys
import subprocess


def history(args):
	args.db_path = DataCollectionService.create_understand_database(args)
	DataCollectionService.compute_and_save_historical_data(args)
	DataCollectionService.fetch_and_save_execution_history(args)
	print(f'All finished, results are saved in {args.output_dir}')


def release(args):
	DataCollectionService.compute_and_save_release_data(args)


def fetch_source_code_if_needed(args):
	path = args.project_path
	slug = args.project_slug
	if path is None and slug is None:
		print(f'At least one of the --project-path or --project-slug should be provided.')
		sys.exit()
	if path is None:
		name = slug.split('/')[-1]
		args.project_path = f'{args.output_dir}/{name}'
		if os.path.isdir(args.project_path):
			return
		clone_command = f'git clone https://github.com/{slug}.git {args.project_path}'
		return_code = subprocess.call(clone_command, shell=True)
		if return_code != 0:
			print('Failure in fetching source code for GitHub!')
			sys.exit()


def valid_date(s):
	try:
		return datetime.strptime(s, "%Y-%m-%d")
	except ValueError:
		msg = "Not a valid date: '{0}'.".format(s)
		raise argparse.ArgumentTypeError(msg)


def add_common_arguments(parser):
	parser.add_argument('-p', '--project-path', help="Project's source code git repository path.", default=None)
	parser.add_argument('-s', '--project-slug', help="The project's GitHub slug, e.g., apache/commons.", default=None)
	parser.add_argument('-t', '--test-path', help="Specifies the relative root directory of the test source code.", required=True)
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
	release_parser.add_argument('-from', '--from-commit', help="Hash of the start commit of this release.", required=True)
	release_parser.add_argument('-to', '--to-commit', help="Hash of the last commit of this release.", required=True)
	release_parser.add_argument('-hist', '--histories-dir', help="Path to outputs of the history command.", required=True)

	args = parser.parse_args()
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	fetch_source_code_if_needed(args)
	args.unique_separator = '\t'
	args.func(args)


if __name__ == "__main__":
	main()
