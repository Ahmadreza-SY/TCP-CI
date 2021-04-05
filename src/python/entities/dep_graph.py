import pandas as pd


class Edge:
    def __init__(self, _from: int, to: int):
        self._from = _from
        self.to = to


class DepGraph:
    def __init__(self):
        self.graph = {}
        self.weights = {}

    def add_edge(self, _from, to):
        self.graph.setdefault(_from, [])
        if to not in self.graph[_from] and _from != to:
            self.graph[_from].append(to)

    def add_weights(self, _from, weights):
        if _from in self.graph and len(self.graph[_from]) == len(weights):
            self.weights[_from] = weights

    def reverse_graph(self):
        reversed_graph = {}
        reversed_weights = {}
        for src, dest_ids in self.graph.items():
            for i, dest in enumerate(dest_ids):
                reversed_graph.setdefault(dest, [])
                reversed_weights.setdefault(dest, [])
                reversed_graph[dest].append(src)
                reversed_weights[dest].append(self.weights[src][i])
        self.graph = reversed_graph
        self.weights = reversed_weights

    def save_graph(self, file_path, dependency_col_name, sep=","):
        graph_df = pd.DataFrame(
            {
                "entity_id": list(self.graph.keys()),
                dependency_col_name: list(self.graph.values()),
            }
        )
        graph_df["weights"] = [self.weights[ent_id] for ent_id in self.graph.keys()]
        graph_df.to_csv(file_path, sep=sep, index=False)
