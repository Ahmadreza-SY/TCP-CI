from entities.entity import Entity
from typing import List

class Edge:
    def __init__(self, _from: Entity, to: Entity):
        self._from = _from
        self.to = to


class DepGraph:
    def __init__(self, edges: List[Edge]):
        self.edges = edges
