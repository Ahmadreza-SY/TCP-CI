from enum import Enum


class EntityType(Enum):
    TEST = 1
    SRC = 2


class Language(Enum):
    JAVA = 1
    C = 1


class Entity:
    def __init__(self, id: int, type: EntityType, rel_path: str, lang: Language, name: str):
        self.id = id
        self.type = type
        self.rel_path = rel_path
        self.lang = lang
        self.name = name


class File(Entity):
    def __init__(self, id, type, rel_path, lang, name, package_name: str):
        Entity.__init__(self, id, type, rel_path, lang, name)
        self.package_name = package_name


class Function(Entity):
    def __init__(self, id, type, rel_path, lang, name, parameters: str, unique_name: str):
        Entity.__init__(self, id, type, rel_path, lang, name)
        self.parameters = parameters
        self.unique_name = unique_name
