from .execution_record_extractor import *
from ..entities.execution_record import ExecutionRecord, TestVerdict, Build
import pandas as pd
from tqdm import tqdm
from ..timer import tik_list, tok_list
import logging


class TorrentExtractor(ExecutionRecordExtractorInterface):
    def __init__(self, config, repository_miner):
        self.config = config
        self.repository_miner = repository_miner

    def create_test_name_to_id_mapping(self, metadata_df):
        package_to_id = dict(zip(metadata_df.Package.values, metadata_df.Id.values))
        return package_to_id

    def fetch_execution_records(self) -> Tuple[List[ExecutionRecord], List[Build]]:
        if self.config.project_slug is None:
            logging.warn(
                f"No project slug is provided, skipping test execution history retrival."
            )
            return [], []
        if self.config.ci_data_path is None:
            logging.warn(
                f"No path for RTP-Torrent data is provided, skipping test execution history retrival."
            )
            return [], []

        logging.info("Reading RTP-Torrent execution records ...")
        user_login, project_name = self.config.project_slug.split("/")
        rtp_exe_df = pd.read_csv(
            self.config.ci_data_path
            / f"{user_login}@{project_name}"
            / f"{user_login}@{project_name}-full.csv"
        )
        selected_jobs = (
            rtp_exe_df.groupby(["travisBuildId", "travisJobId"], as_index=False)
            .count()
            .sort_values(
                ["testName", "travisJobId"], ignore_index=True, ascending=[False, True]
            )
            .groupby("travisBuildId", as_index=False)
            .first()["travisJobId"]
            .unique()
            .tolist()
        )
        rtp_exe_df = (
            rtp_exe_df[rtp_exe_df["travisJobId"].isin(selected_jobs)]
            .copy()
            .reset_index(drop=True)
        )
        builds_df = pd.read_csv(
            self.config.ci_data_path
            / f"{user_login}@{project_name}"
            / f"{user_login}@{project_name}-builds.csv",
            parse_dates=["gh_build_started_at"],
        )

        builds = []
        for _, row in builds_df.iterrows():
            builds.append(
                Build(
                    int(row["tr_build_id"]),
                    row["git_all_built_commits"].split("#"),
                    row["gh_build_started_at"],
                )
            )
        builds.sort(key=lambda b: b.started_at)

        exe_records = []
        for build in tqdm(builds, desc="Creating execution records"):
            metadata_path = (
                self.repository_miner.get_analysis_path(build) / "metadata.csv"
            )
            if not metadata_path.exists():
                result = self.repository_miner.analyze_build_statically(build)
                if result.empty:
                    continue

            tik_list(["REC_P", "Total"], build.id)
            entities_df = pd.read_csv(metadata_path)
            build_exe_df = (
                rtp_exe_df[rtp_exe_df["travisBuildId"] == build.id]
                .copy()
                .reset_index(drop=True)
            )
            if len(build_exe_df) == 0:
                continue

            test_name_to_id = self.create_test_name_to_id_mapping(entities_df)
            build_exe_df[ExecutionRecord.TEST] = build_exe_df["testName"].apply(
                lambda name: test_name_to_id.get(name, None)
            )
            build_exe_df.dropna(subset=[ExecutionRecord.TEST], inplace=True)
            build_exe_df[ExecutionRecord.TEST] = build_exe_df[
                ExecutionRecord.TEST
            ].astype("int32")
            build_exe_df = (
                build_exe_df.groupby(
                    ["travisBuildId", "travisJobId", ExecutionRecord.TEST],
                    as_index=False,
                )
                .agg(
                    {
                        "testName": "first",
                        "duration": sum,
                        "count": sum,
                        "failures": sum,
                        "errors": sum,
                        "skipped": sum,
                    }
                )
                .reset_index(drop=True)
            )

            for _, row in build_exe_df.iterrows():
                test_verdict = TestVerdict.SUCCESS
                if row["failures"] > 0 or row["errors"] > 0:
                    test_verdict = (
                        TestVerdict.ASSERTION
                        if row["failures"] >= row["errors"]
                        else TestVerdict.EXCEPTION
                    )
                exe_record = ExecutionRecord(
                    row[ExecutionRecord.TEST],
                    row["travisBuildId"],
                    row["travisJobId"],
                    test_verdict,
                    int(row[ExecutionRecord.DURATION] * 1000.0),
                )
                exe_records.append(exe_record)
            tok_list(["REC_P", "Total"], build.id)

        return exe_records, builds
