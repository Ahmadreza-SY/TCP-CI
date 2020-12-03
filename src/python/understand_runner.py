import time
import subprocess
import shlex
import os
import sys
from tqdm import tqdm

class UnderstandRunner:

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
			UnderstandRunner.run_und_command(und_command, args)
			print(f'Created understand db at {und_db_path}, took {"{0:.2f}".format(time.time() - start)} seconds.')
		else:
			print(f'Understand db already exists at {und_db_path}, skipping understand analysis.')
		return und_db_path
