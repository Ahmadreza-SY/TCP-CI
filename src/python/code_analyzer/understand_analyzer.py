from code_analyzer import *


class UnderstandAnalyzer(CodeAnalyzerInterface):
    def get_entities() -> List[Entity]:
        pass

    def compute_metrics(entity: Entity) -> Dict[str, str]:
        pass

    def compute_dependency_graph(src: List[Entity], dest: List[Entity]) -> DepGraph:
        pass
