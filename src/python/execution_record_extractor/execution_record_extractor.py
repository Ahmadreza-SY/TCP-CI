from ..entities.execution_record import ExecutionRecord, Build
from typing import List, Tuple


class ExecutionRecordExtractorInterface:
    def fetch_execution_records(self) -> Tuple[List[ExecutionRecord], List[Build]]:
        pass