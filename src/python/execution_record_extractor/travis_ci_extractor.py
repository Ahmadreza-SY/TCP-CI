from .execution_record_extractor import *
from enum import Enum
from ..entities.execution_record import ExecutionRecord, TestVerdict, Build
from ..entities.entity import Language
from ..code_analyzer.code_analyzer import AnalysisLevel
import pandas as pd
import subprocess
import shlex
import os
import re


class LogType(Enum):
    MAVEN = 0
    GTEST = 1


class TravisCIExtractor(ExecutionRecordExtractorInterface):
    SUPPORTED_LANG_LEVELS = [
        (Language.JAVA, AnalysisLevel.FILE),
        (Language.C, AnalysisLevel.FILE),
    ]

    def __init__(
        self, language, level, project_slug, project_path, output_path, unique_separator
    ):
        self.language = language
        self.level = level
        self.project_slug = project_slug
        self.project_path = project_path
        self.output_path = output_path
        self.unique_separator = unique_separator

    def get_log_type(self):
        pass

    def create_test_name_to_id_mapping(exe_df):
        pass

    def fetch_execution_records(self) -> List[ExecutionRecord]:
        if (self.language, self.level) not in TravisCIExtractor.SUPPORTED_LANG_LEVELS:
            print(
                f"Test execution history extraction is not yet supported for {self.language} in {self.level} granularity level."
            )
            return

        if self.project_slug is None:
            print(
                f"No project slug is provided, skipping test execution history retrival."
            )
            return

        if not os.path.exists(f"{self.output_path}/test_execution_history.csv"):
            log_type = self.get_log_type().value
            command = f"ruby ./src/ruby/exe_feature_extractor.rb {self.project_slug} {log_type} {self.output_path}"
            return_code = subprocess.call(shlex.split(command))
            if return_code != 0:
                print(f"failed ruby test execution history command: {command}")
                return
        else:
            print("Execution history exists, skipping fetch.")

        exe_df = pd.read_csv(f"{self.output_path}/test_execution_history.csv")
        test_name_to_id = self.create_test_name_to_id_mapping(exe_df)
        exe_df[ExecutionRecord.TEST] = exe_df["test_name"].apply(
            lambda name: test_name_to_id.get(name, None)
        )
        exe_df.drop(["test_name"], axis=1, inplace=True)
        exe_df.dropna(subset=[ExecutionRecord.TEST], inplace=True)
        exe_df[ExecutionRecord.TEST] = exe_df[ExecutionRecord.TEST].astype("int32")
        exe_df = (
            exe_df.groupby([ExecutionRecord.TEST, "build", "job"]).sum().reset_index()
        )
        exe_df["test_result"] = exe_df["test_result"].apply(
            lambda result: result
            if result <= TestVerdict.UNKNOWN_FAILURE.value
            else TestVerdict.UNKNOWN_FAILURE.value
        )
        exe_df.sort_values(by=["build", "job"], ascending=False, inplace=True)

        exe_records = []
        for i, row in exe_df.iterrows():
            exe_record = ExecutionRecord(
                row[ExecutionRecord.TEST],
                row["build"],
                row["job"],
                TestVerdict(row["test_result"]),
                row["duration"],
            )
            exe_records.append(exe_record)

        builds_df = pd.read_csv(
            f"{self.output_path}/full_builds.csv", sep=self.unique_separator
        )
        builds_df.sort_values(by=["id"], ascending=False, inplace=True)
        builds = []
        for i, row in builds_df.iterrows():
            builds.append(Build(row["id"], row["commit_hash"]))

        return exe_records, builds


class TravisCIJavaExtractor(TravisCIExtractor):
    def get_log_type(self):
        return LogType.MAVEN

    def create_test_name_to_id_mapping(self, exe_df):
        metadata_df = pd.read_csv(
            f"{self.output_path}/metadata.csv", usecols=["Id", "Package"]
        )
        package_to_id = dict(zip(metadata_df.Package.values, metadata_df.Id.values))
        return package_to_id


class TravisCICExtractor(TravisCIExtractor):
    def get_log_type(self):
        return LogType.GTEST

    def create_test_name_to_id_mapping(self, exe_df):
        metadata_df = pd.read_csv(
            f"{self.output_path}/metadata.csv", usecols=["Id", "FilePath", "EntityType"]
        )
        test_metadata = metadata_df[metadata_df.EntityType == "TEST"]
        test_files = zip(test_metadata.Id.values, test_metadata.FilePath.values)
        test_names = exe_df.test_name.unique()
        test_name_to_id = {}
        test_contents = []
        for test_id, test_file in test_files:
            with open(f"{self.project_path}/{test_file}") as f:
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
