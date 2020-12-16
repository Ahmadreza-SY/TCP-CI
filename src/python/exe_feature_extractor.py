import pandas as pd
import subprocess
import shlex
import os


class ExeFeatureExtractor:

	SUPPORTED_LANG_LEVELS = [('java', 'file')]

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

		command = f'ruby ./src/ruby/exe_feature_extractor.rb {args.project_slug} {args.output_dir}'
		return_code = subprocess.call(shlex.split(command))

		if return_code != 0:
			print(f'failed ruby test execution history command: {command}')
			return

		exe_df = pd.read_csv(f'{args.output_dir}/test_execution_history.csv')
		metadata_df = pd.read_csv(f'{args.output_dir}/metadata.csv', usecols=['Id', 'Package'])
		package_to_id = dict(zip(metadata_df.Package.values, metadata_df.Id.values))
		exe_df['entity_id'] = exe_df['test_name'].apply(lambda name: package_to_id.get(name, None))
		exe_df.drop(['test_name'], axis=1, inplace=True)
		exe_df.dropna(subset=['entity_id'], inplace=True)
		exe_df['entity_id'] = exe_df['entity_id'].astype('int32')
		exe_cols = exe_df.columns.tolist()
		exe_cols = exe_cols[-1:] + exe_cols[:-1]
		exe_df = exe_df[exe_cols]
		exe_df.sort_values(by=['build', 'job'], ascending=False, inplace=True)
		exe_df.to_csv(f'{args.output_dir}/exe.csv', index=False)
		os.remove(f'{args.output_dir}/test_execution_history.csv')

		ExeFeatureExtractor.compute_test_case_features(exe_df).to_csv(f'{args.output_dir}/test_cases.csv', index=False)

		builds_df = pd.read_csv(f'{args.output_dir}/builds.csv', sep=';')
		builds_df.sort_values(by=['id'], ascending=False, inplace=True)
		builds_df.to_csv(f'{args.output_dir}/builds.csv', index=False, sep=';')

		commits_df = pd.read_csv(f'{args.output_dir}/commits.csv', usecols=['hash', 'committer', 'author'], sep=';')
		contributors_df = pd.read_csv(f'{args.output_dir}/contributors.csv')
		contributors_df = ExeFeatureExtractor.compute_contributors_failure_rate(exe_df, builds_df, commits_df, contributors_df)
		contributors_df.to_csv(f'{args.output_dir}/contributors.csv', index=False)

		jobs_df = pd.read_csv(f'{args.output_dir}/jobs.csv', sep=';')
		jobs_df.sort_values(by=['build_id', 'id'], ascending=False, inplace=True)
		jobs_df.to_csv(f'{args.output_dir}/jobs.csv', index=False, sep=';')
		return
