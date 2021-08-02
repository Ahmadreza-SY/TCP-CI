import pandas as pd
import re
from tqdm import tqdm


class TrTorrentProcessor:
    TEST_RUN_REGEX = r"Running (([A-Za-z]{1}[A-Za-z\d_]*\.)+[A-Za-z][A-Za-z\d_]*)(.*?)Tests run: (\d*), Failures: (\d*), Errors: (\d*), Skipped: (\d*), Time elapsed: ([+-]?([0-9]*[.])?[0-9]+)"

    def __init__(self):
        pass

    def extract_test_runs(self, log_path):
        log = ""
        with open(log_path) as f:
            log = f.read()

        test_runs = []
        matches = re.finditer(
            TrTorrentProcessor.TEST_RUN_REGEX, log, re.MULTILINE | re.DOTALL
        )
        for match in matches:
            test_run = {}
            test_run["testName"] = match.group(1)
            test_run["duration"] = float(match.group(8))
            test_run["count"] = int(match.group(4))
            test_run["failures"] = int(match.group(5))
            test_run["errors"] = int(match.group(6))
            test_run["skipped"] = int(match.group(7))
            test_runs.append(test_run)
        return test_runs

    def extract_exe_records(self, logs_path):
        records = []
        repo_data_df = pd.read_json(logs_path / "repo-data-travis.json")
        rows = list(repo_data_df.iterrows())
        for _, r in tqdm(rows, desc=f"Processing logs for {logs_path.name}"):
            build_id = r["build_id"]
            for job_id in r["jobs"]:
                log_paths = list(
                    logs_path.glob(f"*_{build_id}_{r['commit']}_{job_id}.log")
                )
                if len(log_paths) > 1:
                    raise Exception(f"No unique log found: {log_paths}")
                log_path = log_paths[0]
                test_runs = self.extract_test_runs(log_path)
                for test_run in test_runs:
                    record = {}
                    record["travisBuildId"] = build_id
                    record["travisJobId"] = job_id
                    record = {**record, **test_run}
                    records.append(record)
        return pd.DataFrame(records)

    def extract_builds(self, logs_path, data_path):
        repo_data_df = pd.read_json(logs_path / "repo-data-travis.json")
        builds_df = pd.read_csv(data_path / "data.csv")
        builds_df = builds_df.merge(
            repo_data_df[["build_id", "jobs"]],
            left_on="tr_build_id",
            right_on="build_id",
        ).drop("build_id", axis=1)
        return builds_df

    def process_tr_torrent_data(self, source_path, output_path, repo):
        print(f"Processing {repo} data")
        logs_path = source_path / "build_logs" / repo
        data_path = source_path / "data" / repo
        repo_output_path = output_path / repo
        exe_output_file = repo_output_path / f"{repo}-full.csv"
        builds_output_file = repo_output_path / f"{repo}-builds.csv"
        if exe_output_file.exists() and builds_output_file.exists():
            return
        if not (
            (data_path / "data.csv").exists()
            and (logs_path / "repo-data-travis.json").exists()
        ):
            return
        exe_df = self.extract_exe_records(logs_path)
        builds_df = self.extract_builds(logs_path, data_path)
        repo_output_path.mkdir(parents=True, exist_ok=True)
        exe_df.to_csv(exe_output_file, index=False)
        builds_df.to_csv(builds_output_file, index=False)
        print(f"Finished processing {repo} data")
