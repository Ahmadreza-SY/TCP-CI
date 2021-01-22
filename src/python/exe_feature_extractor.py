import pandas as pd
import subprocess
import shlex
import os
from enum import Enum
import re


class LogType(Enum):
	MAVEN = 0
	GTEST = 1


class TestResult(Enum):
	SUCCESS = 0
	EXCEPTION = 1
	ASSERTION = 2
	UNKNOWN_FAILURE = 3


class ExeFeatureExtractor:

	SUPPORTED_LANG_LEVELS = [('java', 'file'), ('c', 'file')]

	@staticmethod
	def compute_test_case_features(exe_df):
		def calc_failure_rate(tc_results, row):
			tc_runs = tc_results[(row['entity_id'])]
			tc_passes = 0 if 0 not in tc_runs.index else tc_runs[0]
			return 1 - tc_passes/tc_runs.sum()

		test_cases = pd.DataFrame()
		tc_age = exe_df.groupby('entity_id', as_index=False).count()
		test_cases['entity_id'] = tc_age['entity_id']
		test_cases['age'] = tc_age['test_result']
		tc_avg_duration = exe_df[['entity_id', 'duration']].groupby('entity_id', as_index=False).mean()
		test_cases = pd.merge(test_cases, tc_avg_duration, on=['entity_id'])
		tc_results = exe_df[['entity_id', 'test_result']].groupby(['entity_id', 'test_result'], as_index=False).size()
		test_cases['failure_rate'] = test_cases.apply(lambda r: calc_failure_rate(tc_results, r), axis=1)
		test_cases.rename(columns={'duration': 'avg_duration'}, inplace=True)
		return test_cases

	@staticmethod
	def compute_contributors_failure_rate(exe_df, builds_df, commits_df, contributors_df):
		def update_contirbutor_failure_rate(fr_map, contributor, result):
			if contributor not in fr_map:
				fr_map[contributor] = [0, 0]
			if result > 0:
				fr_map[contributor][1] += 1
			else:
				fr_map[contributor][0] += 1

		builds_df.rename(columns={'id': 'build'}, inplace=True)
		builds_tc_results = exe_df.groupby('build', as_index=False).sum()
		builds_tc_results = pd.merge(builds_tc_results, builds_df, on=['build'])
		commit_to_build_result = dict(zip(builds_tc_results.commit_hash, builds_tc_results.test_result))
		contributor_failure_rate = {}
		for index, commit in commits_df.iterrows():
			if commit.hash not in commit_to_build_result:
				continue
			if commit.committer == commit.author:
				update_contirbutor_failure_rate(contributor_failure_rate, commit.committer, commit_to_build_result[commit.hash])
			else:
				update_contirbutor_failure_rate(contributor_failure_rate, commit.committer, commit_to_build_result[commit.hash])
				update_contirbutor_failure_rate(contributor_failure_rate, commit.author, commit_to_build_result[commit.hash])
		contributor_failure_rate = pd.DataFrame({'id': list(contributor_failure_rate.keys()),
																						 'failure_rate': list(map(lambda r: r[1]/sum(r), contributor_failure_rate.values()))})
		contributors_df = pd.merge(contributors_df, contributor_failure_rate, on=['id'], how='outer')
		contributors_df.fillna(0, inplace=True)
		return contributors_df

	@staticmethod
	def fetch_and_save_execution_history(args):
		if (args.language, args.level) not in ExeFeatureExtractor.SUPPORTED_LANG_LEVELS:
			print(f'Test execution history extraction is not yet supported for {args.language} in {args.level} granularity level.')
			return

		if args.project_slug is None:
			print(f'No project slug is provided, skipping test execution history retrival.')
			return

		log_type = None
		exe_mapper = None
		if args.language == "java":
			log_type = LogType.MAVEN.value
			exe_mapper = JavaFileExeMapper
		elif args.language == "c":
			log_type = LogType.GTEST.value
			exe_mapper = CFileExeMapper

		command = f'ruby ./src/ruby/exe_feature_extractor.rb {args.project_slug} {log_type} {args.output_dir}'
		return_code = subprocess.call(shlex.split(command))
		if return_code != 0:
			print(f'failed ruby test execution history command: {command}')
			return

		exe_df = pd.read_csv(f'{args.output_dir}/test_execution_history.csv')
		test_name_to_id = exe_mapper.compute_test_name_to_id(exe_df, args)
		exe_df['entity_id'] = exe_df['test_name'].apply(lambda name: test_name_to_id.get(name, None))
		exe_df.drop(['test_name'], axis=1, inplace=True)
		exe_df.dropna(subset=['entity_id'], inplace=True)
		exe_df['entity_id'] = exe_df['entity_id'].astype('int32')
		exe_df = exe_df.groupby(['entity_id', 'build', 'job']).sum().reset_index()
		exe_df['test_result'] = exe_df['test_result'].apply(
				lambda result: result if result <= TestResult.UNKNOWN_FAILURE.value else TestResult.UNKNOWN_FAILURE.value)
		exe_df.sort_values(by=['build', 'job'], ascending=False, inplace=True)
		exe_df.to_csv(f'{args.output_dir}/exe.csv', index=False)
		os.remove(f'{args.output_dir}/test_execution_history.csv')

		ExeFeatureExtractor.compute_test_case_features(exe_df).to_csv(f'{args.output_dir}/test_cases.csv', index=False)

		builds_df = pd.read_csv(f'{args.output_dir}/builds.csv', sep=args.unique_separator)
		builds_df.sort_values(by=['id'], ascending=False, inplace=True)
		builds_df.to_csv(f'{args.output_dir}/builds.csv', index=False, sep=args.unique_separator)

		commits_df = pd.read_csv(f'{args.output_dir}/commits.csv', usecols=['hash', 'committer', 'author'], sep=args.unique_separator)
		contributors_df = pd.read_csv(f'{args.output_dir}/contributors.csv')
		contributors_df = ExeFeatureExtractor.compute_contributors_failure_rate(exe_df, builds_df, commits_df, contributors_df)
		contributors_df.to_csv(f'{args.output_dir}/contributors.csv', index=False)

		jobs_df = pd.read_csv(f'{args.output_dir}/jobs.csv', sep=args.unique_separator)
		jobs_df.sort_values(by=['build_id', 'id'], ascending=False, inplace=True)
		jobs_df.to_csv(f'{args.output_dir}/jobs.csv', index=False, sep=args.unique_separator)
		return


class JavaFileExeMapper:

	@staticmethod
	def compute_test_name_to_id(exe_df, args):
		metadata_df = pd.read_csv(f'{args.output_dir}/metadata.csv', usecols=['Id', 'Package'])
		package_to_id = dict(zip(metadata_df.Package.values, metadata_df.Id.values))
		return package_to_id


class CFileExeMapper:

	@staticmethod
	def compute_test_name_to_id(exe_df, args):
		metadata_df = pd.read_csv(f'{args.output_dir}/metadata.csv', usecols=['Id', 'FilePath', 'EntityType'])
		test_metadata = metadata_df[metadata_df.EntityType == "TEST"]
		test_files = zip(test_metadata.Id.values, test_metadata.FilePath.values)
		test_names = exe_df.test_name.unique()
		test_name_to_id = {}
		test_contents = []
		for test_id, test_file in test_files:
			with open(f'{args.project_path}/{test_file}') as f:
				content = re.sub('\s+', '', f.read())
				test_contents.append((test_id, content))
		for test_name in test_names:
			items = test_name.split('.')
			if len(items) != 2:
				continue
			suit, name = items
			suit = re.sub("/\d+", "", suit)
			
			for test_id, test_content in test_contents:
				if re.search(f'(TEST|TEST_F|TYPED_TEST)\({suit},{name}\)', test_content):
					test_name_to_id[test_name] = test_id
					break
		return test_name_to_id
