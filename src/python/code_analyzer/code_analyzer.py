from typing import List, Dict
from entities.entity import Entity
from entities.dep_graph import DepGraph


class CodeAnalyzerInterface:
    def get_entities() -> List[Entity]:
        pass

    def compute_metrics(entity: Entity) -> Dict[str, str]:
        pass

    def compute_dependency_graph(src: List[Entity], dest: List[Entity]) -> DepGraph:
        pass
