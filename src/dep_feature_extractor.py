from pydriller import RepositoryMining
from pydriller.domain.commit import ModificationType
from datetime import datetime
from src.association_miner import *
from src.understand_database import UnderstandDatabase
import pandas as pd
from tqdm import tqdm


class DEPExtractor:
	ID_FIELD = "Id"
	NAME_FIELD = "Name"
	FILE_PATH_FIELD = "FilePath"

	def __init__(self, understand_db, lang):
		self.understand_db = understand_db
		self.lang = lang

	def get_association_miner(self):
		pass

	def extract_metadata(self):
		pass

	def extract_structural_dependency_graph(self, metadata):
		structural_graph = {}
		entities = self.understand_db.get_ents_by_id(metadata.Id.values)
		ent_id_set = set(metadata.Id.values)
		for entity in tqdm(entities, desc="Extracting structural graph"):
			for dep in self.understand_db.get_dependencies(entity):
				if dep.id() in ent_id_set:
					structural_graph.setdefault(entity.id(), [])
					if dep.id() not in structural_graph[entity.id()] and dep.id() != entity.id():
						structural_graph[entity.id()].append(dep.id())
		return structural_graph

	def extract_logical_dependency_graph(self, structural_graph, association_map):
		logical_graph = {}
		for edge_start, dependencies in tqdm(structural_graph.items(), desc="Extracting edge weights"):
			dep_weights = []
			for edge_end in dependencies:
				pair = frozenset({edge_start, edge_end})
				if pair in association_map:
					rule = association_map[pair]
					forward_stats = next(stats for stats in rule.ordered_statistics if stats.items_base == frozenset({edge_end}))
					backward_stats = next(stats for stats in rule.ordered_statistics if stats.items_base == frozenset({edge_start}))
					dep_weights.append([rule.support, forward_stats.confidence, backward_stats.confidence, forward_stats.lift])
				else:
					dep_weights.append(0)
			logical_graph[edge_start] = dep_weights
		return logical_graph


class FileDEPExtractor(DEPExtractor):
	def get_association_miner(self):
		return FileAssociationMiner

	def extract_metadata(self):
		metadata = {}
		metadata.setdefault(DEPExtractor.ID_FIELD, [])
		metadata.setdefault(DEPExtractor.NAME_FIELD, [])
		metadata.setdefault(DEPExtractor.FILE_PATH_FIELD, [])
		entities = self.understand_db.get_ents()
		for entity in tqdm(entities, desc="Extracting metadata"):
			if not self.understand_db.entity_is_valid(entity):
				continue
			metadata[DEPExtractor.ID_FIELD].append(entity.id())
			metadata[DEPExtractor.NAME_FIELD].append(entity.name())
			rel_path = self.understand_db.get_valid_rel_path(entity)
			metadata[DEPExtractor.FILE_PATH_FIELD].append(rel_path)
			metric_names = entity.metrics()
			metrics = entity.metric(metric_names)
			for name, value in metrics.items():
				metadata.setdefault(name, [])
				metadata[name].append(value)
		return metadata


class FunctionDEPExtractor(DEPExtractor):
	FULL_NAME_FIELD = "FullName"
	UNIQUE_NAME_FIELD = "UniqueName"
	PARAMETERS_FIELD = "Parameters"

	def get_association_miner(self):
		return FunctionAssociationMiner

	def extract_metadata(self):
		metadata = {}
		metadata.setdefault(DEPExtractor.ID_FIELD, [])
		metadata.setdefault(DEPExtractor.NAME_FIELD, [])
		metadata.setdefault(FunctionDEPExtractor.FULL_NAME_FIELD, [])
		metadata.setdefault(FunctionDEPExtractor.UNIQUE_NAME_FIELD, [])
		metadata.setdefault(DEPExtractor.FILE_PATH_FIELD, [])
		metadata.setdefault(FunctionDEPExtractor.PARAMETERS_FIELD, [])
		entities = self.understand_db.get_ents()
		function_set = set()
		for entity in tqdm(entities, desc="Extracting metadata"):
			if not self.understand_db.entity_is_valid(entity):
				continue
			rel_path = self.understand_db.get_valid_rel_path(entity.ref('definein').file())
			parameters = "" if not entity.parameters() else entity.parameters()
			unique_name = f'{entity.name()}-{entity.longname()}-{rel_path}-{parameters}'
			if unique_name in function_set:
				continue
			function_set.add(unique_name)
			metadata[DEPExtractor.ID_FIELD].append(entity.id())
			metadata[DEPExtractor.NAME_FIELD].append(entity.name())
			metadata[FunctionDEPExtractor.FULL_NAME_FIELD].append(entity.longname())
			metadata[FunctionDEPExtractor.UNIQUE_NAME_FIELD].append(self.understand_db.get_und_function_unique_name(entity))
			metadata[DEPExtractor.FILE_PATH_FIELD].append(rel_path)
			metadata[FunctionDEPExtractor.PARAMETERS_FIELD].append(parameters)
			metric_names = entity.metrics()
			metrics = entity.metric(metric_names)
			for name, value in metrics.items():
				metadata.setdefault(name, [])
				metadata[name].append(value)
		return metadata
