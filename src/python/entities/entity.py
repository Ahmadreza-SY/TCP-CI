from enum import Enum
from typing import Dict


class EntityType(Enum):
    TEST = 1
    SRC = 2


class Language(Enum):
    JAVA = "java"
    C = "c"


class Entity:
    ID = "Id"
    NAME = "Name"
    ENTITY_TYPE = "EntityType"
    FILE_PATH = "FilePath"

    def __init__(
        self,
        id: int,
        type: EntityType,
        rel_path: str,
        lang: Language,
        name: str,
        metrics: Dict[str, str],
    ):
        self.id = id
        self.type = type
        self.rel_path = rel_path
        self.lang = lang
        self.name = name
        self.metrics = metrics

    def to_dict(self):
        d = {
            Entity.ID: self.id,
            Entity.NAME: self.name,
            Entity.ENTITY_TYPE: self.type.name,
            Entity.FILE_PATH: self.rel_path,
        }
        for name, value in self.metrics.items():
            d[name] = value
        return d


class File(Entity):
    PACKAGE = "Package"

    def __init__(self, id, type, rel_path, lang, name, metrics, package_name: str):
        Entity.__init__(self, id, type, rel_path, lang, name, metrics)
        self.package_name = package_name

    def to_dict(self):
        entity_dict = super().to_dict()
        entity_dict[File.PACKAGE] = self.package_name
        return entity_dict


class Function(Entity):
    PARAMETERS = "Parameters"
    UNIQUE_NAME = "UniqueName"

    def __init__(
        self, id, type, rel_path, lang, name, metrics, parameters: str, unique_name: str
    ):
        Entity.__init__(self, id, type, rel_path, lang, name, metrics)
        self.parameters = parameters
        self.unique_name = unique_name

    def to_dict(self):
        entity_dict = super().to_dict()
        entity_dict[Function.PARAMETERS] = self.parameters
        entity_dict[Function.UNIQUE_NAME] = self.unique_name
        return entity_dict
