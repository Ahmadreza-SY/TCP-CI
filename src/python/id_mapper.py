import pandas as pd
from pathlib import Path
import numpy as np


class IdMapper:
    def __init__(self, output_path: Path):
        self.id_map_path = output_path / "id_map.csv"
        if self.id_map_path.exists():
            id_map_df = pd.read_csv(self.id_map_path)
            self.id_map = dict(
                zip(id_map_df.key.values.tolist(), id_map_df.value.values.tolist())
            )
            self.max_id = np.max(id_map_df.value.values)
        else:
            self.id_map = {}
            self.max_id = 0

    def get_entity_id(self, unique_identifier):
        id = -1
        if unique_identifier in self.id_map:
            id = self.id_map[unique_identifier]
        else:
            self.max_id += 1
            self.id_map[unique_identifier] = self.max_id
            id = self.max_id
        return id

    def merge_entity_ids(self, unique_identifiers):
        a = unique_identifiers[0]
        b = unique_identifiers[1]
        if a in self.id_map:
            self.id_map[b] = self.id_map[a]
        elif a not in self.id_map and b in self.id_map:
            self.id_map[a] = self.id_map[b]
        else:
            self.get_entity_id(a)
            self.id_map[b] = self.id_map[a]
        return self.id_map[a]

    def save_id_map(self):
        id_map_df = pd.DataFrame(
            {"key": list(self.id_map.keys()), "value": list(self.id_map.values())}
        )
        id_map_df.to_csv(self.id_map_path, index=False)
