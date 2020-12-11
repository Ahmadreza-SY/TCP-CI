import pandas as pd
import subprocess
import shlex
import os


class ExeFeatureExtractor:

	SUPPORTED_LANG_LEVELS = [('java', 'file')]

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

		builds_df = pd.read_csv(f'{args.output_dir}/builds.csv', sep=';')
		builds_df.sort_values(by=['id'], ascending=False, inplace=True)
		builds_df.to_csv(f'{args.output_dir}/builds.csv', index=False, sep=';')

		jobs_df = pd.read_csv(f'{args.output_dir}/jobs.csv', sep=';')
		jobs_df.sort_values(by=['build_id', 'id'], ascending=False, inplace=True)
		jobs_df.to_csv(f'{args.output_dir}/jobs.csv', index=False, sep=';')
		return
