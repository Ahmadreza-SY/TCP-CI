from entities.execution_record import ExecutionRecord
from typing import List


class ExecutionRecordExtractorInterface:
    def fetch_execution_records(self) -> List[ExecutionRecord]:
        pass
