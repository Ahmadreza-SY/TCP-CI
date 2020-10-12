
class DEPExtractor:
	ID_FIELD = "Id"
	NAME_FIELD = "Name"
	FILE_PATH_FIELD = "FilePath"

	def __init__(self, db, lang):
		self.db = db
		self.lang = lang

	def get_ents(self):
		pass

	def get_dependencies(self, entity):
		pass

	def extract_metadata(self):
		metadata = {}
		metadata.setdefault(DEPExtractor.ID_FIELD, [])
		metadata.setdefault(DEPExtractor.NAME_FIELD, [])
		entities = self.get_ents()
		for entity in entities:
			metadata[DEPExtractor.ID_FIELD].append(entity.id())
			metadata[DEPExtractor.NAME_FIELD].append(entity.name())
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
					if ref.ent().id() not in graph[entity.id()]:
						graph[entity.id()].append(ref.ent().id())
		return graph


class FileDEPExtractor(DEPExtractor):
	def get_ents(self):
		return self.db.ents(f"{self.lang} file ~unresolved ~unknown")

	def get_dependencies(self, entity):
		return entity.refs(f'{self.lang} include')

	def extract_metadata(self):
		metadata = DEPExtractor.extract_metadata(self)
		metadata.setdefault(DEPExtractor.FILE_PATH_FIELD, [])
		ids = metadata[DEPExtractor.ID_FIELD]
		for id in ids:
			entity = self.db.ent_from_id(id)
			metadata[DEPExtractor.FILE_PATH_FIELD].append(entity.relname())
		return metadata


class FunctionDEPExtractor(DEPExtractor):
	FULL_NAME_FIELD = "FullName"
	PARAMETERS_FIELD = "Parameters"

	def get_ents(self):
		return self.db.ents(f"{self.lang} function ~unresolved ~unknown")

	def get_dependencies(self, entity):
		return entity.refs(f'{self.lang} call')

	def extract_metadata(self):
		metadata = DEPExtractor.extract_metadata(self)
		metadata.setdefault(DEPExtractor.FILE_PATH_FIELD, [])
		metadata.setdefault(FunctionDEPExtractor.FULL_NAME_FIELD, [])
		metadata.setdefault(FunctionDEPExtractor.PARAMETERS_FIELD, [])
		ids = metadata[DEPExtractor.ID_FIELD]
		for id in ids:
			entity = self.db.ent_from_id(id)
			metadata[FunctionDEPExtractor.FULL_NAME_FIELD].append(entity.longname())
			define_in_ref = entity.ref("definein")
			metadata[DEPExtractor.FILE_PATH_FIELD].append(
					None if define_in_ref is None else define_in_ref.file().relname()
			)
			parameters = entity.parameters(False)
			metadata[FunctionDEPExtractor.PARAMETERS_FIELD].append(None if not parameters else parameters)
		return metadata
