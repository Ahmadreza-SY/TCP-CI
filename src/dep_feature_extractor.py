from pydriller import RepositoryMining
from pydriller.domain.commit import ModificationType
from datetime import datetime
from src.association_miner import *
import pandas as pd
from tqdm import tqdm


class DEPExtractor:
	ID_FIELD = "Id"
	NAME_FIELD = "Name"
	FILE_PATH_FIELD = "FilePath"

	def __init__(self, understand_db, lang):
		self.understand_db = understand_db
		self.lang = lang

	def get_ents(self):
		pass

	def get_dependencies(self, entity):
		pass

	def get_association_miner(self):
		pass

	def extract_metadata(self):
		pass

	def extract_structural_dependency_graph(self, metadata):
		structural_graph = {}
		entities = self.get_ents()
		ent_id_set = set(metadata.Id.values)
		for entity in tqdm(entities, desc="Extracting structural graph"):
			for ref in self.get_dependencies(entity):
				if ref.ent().id() in ent_id_set:
					structural_graph.setdefault(entity.id(), [])
					if ref.ent().id() not in structural_graph[entity.id()] and ref.ent().id() != entity.id():
						structural_graph[entity.id()].append(ref.ent().id())
		return structural_graph

	def extract_logical_dependency_graph(self, structural_graph, association_map):
		logical_graph = {}
		for edge_start, dependencies in tqdm(structural_graph.items(), desc="Extracting logical graph"):
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
	def get_ents(self):
		return self.understand_db.ents(f"{self.lang} file ~unresolved ~unknown")

	def get_dependencies(self, entity):
		return entity.refs(f'{self.lang} include')

	def get_association_miner(self):
		return FileAssociationMiner

	def extract_metadata(self):
		metadata = {}
		metadata.setdefault(DEPExtractor.ID_FIELD, [])
		metadata.setdefault(DEPExtractor.NAME_FIELD, [])
		metadata.setdefault(DEPExtractor.FILE_PATH_FIELD, [])
		entities = self.get_ents()
		for entity in tqdm(entities, desc="Extracting metadata"):
			if "/_deps/" in entity.relname():
				continue
			metadata[DEPExtractor.ID_FIELD].append(entity.id())
			metadata[DEPExtractor.NAME_FIELD].append(entity.name())
			metadata[DEPExtractor.FILE_PATH_FIELD].append(entity.relname())
			metric_names = entity.metrics()
			metrics = entity.metric(metric_names)
			for name, value in metrics.items():
				metadata.setdefault(name, [])
				metadata[name].append(value)
		return metadata


class FunctionDEPExtractor(DEPExtractor):
	FULL_NAME_FIELD = "FullName"
	PARAMETERS_FIELD = "Parameters"

	def get_ents(self):
		return self.understand_db.ents(f"{self.lang} function ~unresolved ~unknown")

	def get_dependencies(self, entity):
		return entity.refs(f'{self.lang} call')

	def get_association_miner(self):
		return FunctionAssociationMiner

	def extract_metadata(self):
		metadata = {}
		metadata.setdefault(DEPExtractor.ID_FIELD, [])
		metadata.setdefault(DEPExtractor.NAME_FIELD, [])
		metadata.setdefault(FunctionDEPExtractor.FULL_NAME_FIELD, [])
		metadata.setdefault(DEPExtractor.FILE_PATH_FIELD, [])
		metadata.setdefault(FunctionDEPExtractor.PARAMETERS_FIELD, [])
		entities = self.get_ents()
		for entity in tqdm(entities, desc="Extracting metadata"):
			define_in_ref = entity.ref("definein")
			if define_in_ref is None or "/_deps/" in define_in_ref.file().relname():
				continue
			metadata[DEPExtractor.ID_FIELD].append(entity.id())
			metadata[DEPExtractor.NAME_FIELD].append(entity.name())
			metadata[FunctionDEPExtractor.FULL_NAME_FIELD].append(entity.longname())
			metadata[DEPExtractor.FILE_PATH_FIELD].append(define_in_ref.file().relname())
			parameters = entity.parameters(False)
			metadata[FunctionDEPExtractor.PARAMETERS_FIELD].append(None if not parameters else parameters)
			metric_names = entity.metrics()
			metrics = entity.metric(metric_names)
			for name, value in metrics.items():
				metadata.setdefault(name, [])
				metadata[name].append(value)
		return metadata
