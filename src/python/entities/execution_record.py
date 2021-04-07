from enum import Enum
from ..entities.entity import Entity


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
    def __init__(self, id: int, commit_hash: str):
        self.id = id
        self.commit_hash = commit_hash

    def to_dict(self):
        d = {"id": self.id, "commit_hash": self.commit_hash}
        return d