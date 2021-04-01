from enum import Enum
from entities.entity import Entity


class TestVerdict(Enum):
    SUCCESS = 0
    EXCEPTION = 1
    ASSERTION = 2
    UNKNOWN_FAILURE = 3


class ExecutionRecord:
    def __init__(self, test: Entity, build_id: int, job_id: int, verdict: TestVerdict, duration: int):
        self.test = test
        self.build_id = build_id
        self.job_id = job_id
        self.verdict = verdict
        self.duration = duration
