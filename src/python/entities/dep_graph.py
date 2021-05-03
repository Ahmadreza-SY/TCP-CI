import pandas as pd
import json


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

    def save_graph(self, file_path, sep=","):
        graph_df = pd.DataFrame(
            {
                "entity_id": list(self.graph.keys()),
                "dependencies": list(self.graph.values()),
            }
        )
        graph_df["weights"] = [self.weights[ent_id] for ent_id in self.graph.keys()]
        graph_df.to_csv(file_path, sep=sep, index=False)

    def load_graph(self, file_path, sep=","):
        graph_df = pd.read_csv(
            file_path,
            sep=sep,
            converters={"dependencies": json.loads, "weights": json.loads},
        )
        self.graph = dict(
            zip(
                graph_df["entity_id"].values.tolist(),
                graph_df["dependencies"].values.tolist(),
            )
        )
        self.weights = dict(
            zip(
                graph_df["entity_id"].values.tolist(),
                graph_df["weights"].values.tolist(),
            )
        )
