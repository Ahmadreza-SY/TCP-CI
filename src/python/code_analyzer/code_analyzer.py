from typing import List
from ..entities.entity import Entity
from ..entities.dep_graph import DepGraph
from enum import Enum


class AnalysisLevel(Enum):
    FILE = "file"
    FUNCTION = "function"

    def __str__(self):
        return self.value


class CodeAnalyzerInterface:
    def get_entities(self) -> List[Entity]:
        pass

    def compute_dependency_graph(
        self, src: List[Entity], dest: List[Entity]
    ) -> DepGraph:
        pass
