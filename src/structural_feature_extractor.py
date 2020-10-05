
class StructuralFeatureExtractor:
	def __init__(self, db, lang):
		self.db = db
		self.lang = lang

	def get_ents(self):
		pass

	def get_dependencies(self, entity):
		pass

	def extract_metadata(self):
		metadata = {}
		metadata.setdefault("id", [])
		metadata.setdefault("name", [])
		entities = self.get_ents()
		for entity in entities:
			metadata["id"].append(entity.id())
			metadata["name"].append(entity.name())
			metric_names = entity.metrics()
			metrics = entity.metric(metric_names)
			for name, value in metrics.items():
				metadata.setdefault(name, [])
				metadata[name].append(value)
		return metadata

	def extract_dependency_graph(self):
		graph = {}
		entities = self.get_ents()
		ent_id_set = set(map(lambda e: e.id(), entities))
		for entity in entities:
			for ref in self.get_dependencies(entity):
				if ref.ent().id() in ent_id_set:
					graph.setdefault(entity.id(), [])
					graph[entity.id()].append(ref.ent().id())
		return graph


class FileStructuralFeatureExtractor(StructuralFeatureExtractor):
	def get_ents(self):
		return self.db.ents(f"{self.lang} file ~unresolved ~unknown")

	def get_dependencies(self, entity):
		return entity.refs(f'{self.lang} include')

	def extract_metadata(self):
		metadata = StructuralFeatureExtractor.extract_metadata(self)
		metadata.setdefault("file_path", [])
		ids = metadata["id"]
		for id in ids:
			entity = self.db.ent_from_id(id)
			metadata["file_path"].append(entity.relname())
		return metadata


class FunctionStructuralFeatureExtractor(StructuralFeatureExtractor):
	def get_ents(self):
		return self.db.ents(f"{self.lang} function ~unresolved ~unknown")

	def get_dependencies(self, entity):
		return entity.refs(f'{self.lang} call')

	def extract_metadata(self):
		metadata = StructuralFeatureExtractor.extract_metadata(self)
		metadata.setdefault("file_path", [])
		metadata.setdefault("full_name", [])
		metadata.setdefault("parameters", [])
		ids = metadata["id"]
		for id in ids:
			entity = self.db.ent_from_id(id)
			metadata["full_name"].append(entity.longname())
			define_in_ref = entity.ref("definein")
			metadata["file_path"].append(None if define_in_ref is None else define_in_ref.file().relname())
			parameters = entity.parameters(False)
			metadata["parameters"].append(None if not parameters else parameters)
		return metadata
