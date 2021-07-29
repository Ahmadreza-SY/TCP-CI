from enum import Enum
from typing import List
from datetime import datetime


class TestVerdict(Enum):
    SUCCESS = 0
    EXCEPTION = 1
    ASSERTION = 2
    UNKNOWN_FAILURE = 3


class ExecutionRecord:
    TEST = "test"
    BUILD = "build"
    JOB = "job"
    VERDICT = "verdict"
    DURATION = "duration"

    def __init__(
        self, test: int, build: int, job: str, verdict: TestVerdict, duration: int
    ):
        self.test = int(test)
        self.build = int(build)
        self.job = job
        self.verdict = verdict
        self.duration = int(duration)

    def to_dict(self):
        d = {
            ExecutionRecord.TEST: self.test,
            ExecutionRecord.BUILD: self.build,
            ExecutionRecord.JOB: self.job,
            ExecutionRecord.VERDICT: self.verdict.value,
            ExecutionRecord.DURATION: self.duration,
        }
        return d


class Build:
    def __init__(self, id: int, commits: List[str], started_at: datetime):
        self.id = id
        self.commits = commits
        self.started_at = started_at

    def to_dict(self):
        return {
            "id": self.id,
            "commits": "#".join(self.commits),
            "started_at": self.started_at,
        }