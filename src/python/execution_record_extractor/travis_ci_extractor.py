from .execution_record_extractor import *
from enum import Enum
from ..entities.execution_record import ExecutionRecord, TestVerdict, Build
from ..entities.entity import Language, EntityType
from ..code_analyzer.code_analyzer import AnalysisLevel
import pandas as pd
import subprocess
import shlex
import re
from tqdm import tqdm
from ..timer import tik_list, tok_list


class LogType(Enum):
    MAVEN = 0
    GTEST = 1


class TravisCIExtractor(ExecutionRecordExtractorInterface):
    SUPPORTED_LANG_LEVELS = [
        (Language.JAVA, AnalysisLevel.FILE),
        (Language.C, AnalysisLevel.FILE),
    ]

    def __init__(self, config, repository_miner):
        self.config = config
        self.repository_miner = repository_miner

    def get_log_type(self):
        pass

    def create_test_name_to_id_mapping(self, exe_df, metadata_df):
        pass

    def fetch_execution_records(self) -> Tuple[List[ExecutionRecord], List[Build]]:
        if (
            self.config.language,
            self.config.level,
        ) not in TravisCIExtractor.SUPPORTED_LANG_LEVELS:
            print(
                f"Test execution history extraction is not yet supported for {self.config.language} in {self.config.level} granularity level."
            )
            return [], []

        if self.config.project_slug is None:
            print(
                f"No project slug is provided, skipping test execution history retrival."
            )
            return [], []

        test_exe_history_path = self.config.output_path / "test_execution_history.csv"
        if not test_exe_history_path.exists():
            log_type = self.get_log_type().value
            command = f"ruby -W0 ./src/ruby/exe_feature_extractor.rb {self.config.project_slug} {log_type} {self.config.output_path.as_posix()}"
            return_code = subprocess.call(shlex.split(command))
            if return_code != 0:
                print(f"failed ruby test execution history command: {command}")
                return [], []
        else:
            print("Execution history exists, skipping fetch.")

        builds_df = pd.read_csv(
            self.config.output_path / "full_builds.csv",
            sep=self.config.unique_separator,
            parse_dates=["start_time"],
        )
        builds_df.sort_values(by=["id"], inplace=True, ignore_index=True)
        builds = []
        for i, row in builds_df.iterrows():
            builds.append(Build(row["id"], [row["commit_hash"]], row["start_time"]))
        exe_df = pd.read_csv(test_exe_history_path, dtype={ExecutionRecord.JOB: str})
        exe_records = []
        full_exe_records = []

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
            build_exe_df = exe_df[(exe_df[ExecutionRecord.BUILD] == build.id)]
            if len(build_exe_df) == 0:
                continue

            test_name_to_id = self.create_test_name_to_id_mapping(
                build_exe_df, entities_df
            )
            build_exe_df[ExecutionRecord.TEST] = build_exe_df["test_name"].apply(
                lambda name: test_name_to_id.get(name, None)
            )
            build_exe_df.drop(["test_name"], axis=1, inplace=True)
            build_exe_df.dropna(subset=[ExecutionRecord.TEST], inplace=True)
            build_exe_df[ExecutionRecord.TEST] = build_exe_df[
                ExecutionRecord.TEST
            ].astype("int32")
            build_exe_df = (
                build_exe_df.groupby(
                    [ExecutionRecord.TEST, ExecutionRecord.BUILD, ExecutionRecord.JOB]
                )
                .sum()
                .reset_index()
            )
            build_exe_df["test_result"] = build_exe_df["test_result"].apply(
                lambda result: result
                if result <= TestVerdict.UNKNOWN_FAILURE.value
                else TestVerdict.UNKNOWN_FAILURE.value
            )
            for i, row in build_exe_df.iterrows():
                exe_record = ExecutionRecord(
                    row[ExecutionRecord.TEST],
                    row[ExecutionRecord.BUILD],
                    row[ExecutionRecord.JOB],
                    TestVerdict(row["test_result"]),
                    row[ExecutionRecord.DURATION],
                )
                if exe_record.job.endswith(".1"):
                    exe_records.append(exe_record)
                full_exe_records.append(exe_record)
            tok_list(["REC_P", "Total"], build.id)

        full_exe_df = pd.DataFrame.from_records([e.to_dict() for e in full_exe_records])
        if len(full_exe_df) > 0:
            full_exe_df.sort_values(
                by=[ExecutionRecord.BUILD, ExecutionRecord.JOB],
                ignore_index=True,
                inplace=True,
            )
            full_exe_df.to_csv(self.config.output_path / "full_exe.csv", index=False)
        return exe_records, builds


class TravisCIJavaExtractor(TravisCIExtractor):
    def get_log_type(self):
        return LogType.MAVEN

    def create_test_name_to_id_mapping(self, exe_df, metadata_df):
        package_to_id = dict(zip(metadata_df.Package.values, metadata_df.Id.values))
        return package_to_id


class TravisCICExtractor(TravisCIExtractor):
    def get_log_type(self):
        return LogType.GTEST

    def create_test_name_to_id_mapping(self, exe_df, metadata_df):
        test_metadata = metadata_df[metadata_df.EntityType == EntityType.TEST.name]
        test_files = zip(test_metadata.Id.values, test_metadata.FilePath.values)
        test_names = exe_df.test_name.unique()
        test_name_to_id = {}
        test_contents = []
        for test_id, test_file in test_files:
            with open(f"{self.config.project_path}/{test_file}") as f:
                content = re.sub("\s+", "", f.read())
                test_contents.append((test_id, content))
        for test_name in test_names:
            items = test_name.split(".")
            if len(items) != 2:
                continue
            suit, name = items
            suit = re.sub("/\d+", "", suit)

            for test_id, test_content in test_contents:
                if re.search(
                    f"(TEST|TEST_F|TYPED_TEST)\({suit},{name}\)", test_content
                ):
                    test_name_to_id[test_name] = test_id
                    break
        return test_name_to_id
