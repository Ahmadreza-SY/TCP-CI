import re
from tqdm import tqdm
from ..code_analyzer import *
from ..understand.understand_database import (
    UnderstandCDatabase,
    UnderstandJavaDatabase,
)
from ...entities.entity import Language, File, Function, EntityType
from ...entities.dep_graph import DepGraph
import pandas as pd
import numpy as np
from pathlib import Path
from ...id_mapper import IdMapper


class UnderstandAnalyzer(CodeAnalyzerInterface):
    def __init__(self, config):
        self.config = config
        if config.language == Language.C:
            self.und_db = UnderstandCDatabase(config)
        elif config.language == Language.JAVA:
            self.und_db = UnderstandJavaDatabase(config)
        self.id_mapper = IdMapper(config.output_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.und_db.db is not None:
            self.und_db.db.close()

    def get_entities(self) -> List[Entity]:
        pass

    def get_unique_identifier(self, und_entity):
        pass

    def compute_dependency_graph(self, src: List[int], dest: List[int]) -> DepGraph:
        dep_graph = DepGraph()
        und_entities = self.und_db.get_ents_by_id(src)
        dest_id_set = set(dest)
        for und_entity in tqdm(und_entities, desc="Computing dependecy graph"):
            for dep in self.und_db.get_dependencies(und_entity):
                if dep.id() in dest_id_set:
                    dep_graph.add_edge(und_entity.id(), dep.id())
        return dep_graph


class UnderstandFileAnalyzer(UnderstandAnalyzer):
    def get_unique_identifier(self, und_entity):
        return str(self.und_db.get_valid_rel_path(und_entity))

    def get_entities(self) -> List[Entity]:
        entities = []
        und_entities = self.und_db.get_ents()
        for und_entity in und_entities:
            if not self.und_db.entity_is_valid(und_entity):
                continue
            id = self.id_mapper.get_entity_id(self.get_unique_identifier(und_entity))
            name = und_entity.name()
            package = None
            if self.config.language == Language.JAVA:
                matches = re.compile("package (.+);").findall(und_entity.contents())
                if len(matches) == 1:
                    package = matches[0] + "." + und_entity.name()[:-5]
            rel_path = self.und_db.get_valid_rel_path(und_entity)
            entity_type = self.und_db.get_entity_type(und_entity, rel_path)
            metric_names = und_entity.metrics()
            metrics = und_entity.metric(metric_names)
            entity = File(
                id, entity_type, rel_path, self.config.language, name, metrics, package
            )
            entities.append(entity)
        self.id_mapper.save_id_map()
        return entities


class UnderstandFunctionAnalyzer(UnderstandAnalyzer):
    def get_unique_identifier(self, und_entity):
        rel_path = self.und_db.get_valid_rel_path(und_entity)
        parameters = "" if not und_entity.parameters() else und_entity.parameters()
        return f"{und_entity.name()}-{und_entity.longname()}-{rel_path}-{parameters}"

    def get_entities(self) -> List[Entity]:
        entities = []
        und_entities = self.und_db.get_ents()
        function_set = set()
        for und_entity in und_entities:
            if not self.und_db.entity_is_valid(und_entity):
                continue
            rel_path = self.und_db.get_valid_rel_path(und_entity.ref("definein").file())
            entity_type = self.und_db.get_entity_type(und_entity, rel_path)
            parameters = "" if not und_entity.parameters() else und_entity.parameters()
            unique_name = (
                f"{und_entity.name()}-{und_entity.longname()}-{rel_path}-{parameters}"
            )
            if unique_name in function_set:
                continue
            if (
                self.config.language == Language.C
                and entity_type == EntityType.TEST
                and und_entity.name() != "TestBody"
            ):
                continue
            function_set.add(unique_name)
            id = self.id_mapper.get_entity_id(self.get_unique_identifier(und_entity))
            name = und_entity.name()
            unique_name = self.und_db.get_und_function_unique_name(und_entity)
            metric_names = und_entity.metrics()
            metrics = und_entity.metric(metric_names)
            entity = Function(
                id,
                entity_type,
                rel_path,
                self.config.language,
                name,
                metrics,
                parameters,
                unique_name,
            )
            entities.append(entity)
        self.id_mapper.save_id_map()
        return entities
