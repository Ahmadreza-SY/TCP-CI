import os
import re
import glob
from enum import Enum


class EntityType(Enum):
	TEST = 1
	SRC = 2


class UnderstandDatabase:
	def __init__(self, db, level, project_path, test_path):
		self.db = db
		self.level = level
		self.project_path = project_path
		self.test_path = test_path

	def get_valid_rel_path(self, entity):
		full_project_path = os.path.abspath(self.project_path)
		full_path = entity.longname()
		if full_project_path not in full_path and "RELATIVE:/" in full_path:
			full_path = full_path.replace("RELATIVE:/", '')
			full_path = '/'.join(full_path.split('/')[1:])
		return full_path.replace(full_project_path + '/', '')

	def get_entity_type(self, entity, rel_path):
		file_name = rel_path.split('/')[-1]
		full_test_path = os.path.abspath(self.test_path)
		pattern = f'{full_test_path}/**/{file_name}'
		for match in glob.glob(pattern, recursive=True):
			if os.path.isfile(match) and rel_path in match:
				return EntityType.TEST
		return EntityType.SRC

	def get_ents_by_id(self, ids):
		return list(map(lambda id: self.db.ent_from_id(id), ids))

	def get_ents(self):
		return self.db.ents(f"{self.lang} {self.level} ~unresolved ~unknown")

	def get_dependencies(self, entity):
		pass

	def get_und_function_unique_name(self, und_function):
		pass

	def get_pydriller_function_unique_name(self, und_function):
		pass

	def entity_is_valid(self, entity):
		entity_file_path = None
		if self.level == 'file':
			entity_file_path = self.get_valid_rel_path(entity)
		elif self.level == 'function':
			define_ref = entity.ref('definein')
			if define_ref is None:
				return False
			entity_file_path = self.get_valid_rel_path(define_ref.file())
		if entity_file_path is not None and entity_file_path[0] == "/":
			return False
		return True


class CUnderstandDatabase(UnderstandDatabase):
	def __init__(self, db, level, project_path, test_path):
		UnderstandDatabase.__init__(self, db, level, project_path, test_path)
		self.lang = "c"

	def get_dependencies(self, entity):
		if self.level == "file":
			return list(entity.depends().keys())
		elif self.level == "function":
			return list(map(lambda ref: ref.ent(), entity.refs('c call')))

	def get_und_function_unique_name(self, und_function):
		return und_function.longname()

	def get_pydriller_function_unique_name(self, pydriller_function):
		function_name = pydriller_function.name
		if function_name == 'TEST':
			function_name = pydriller_function.long_name
			test_names = function_name[function_name.find("(")+1:function_name.find(")")].replace(' ', '').split(',')
			function_name = f'{test_names[0]}_{test_names[1]}_Test::TestBody'
		return function_name

	def entity_is_valid(self, entity):
		if not UnderstandDatabase.entity_is_valid(self, entity):
			return False
		if self.level == "file" and "/_deps/" in entity.relname():
			return False
		elif self.level == "function":
			if entity.name() == "[unnamed]":
				return False
			define_in_ref = entity.ref("definein")
			if define_in_ref is None or "/_deps/" in define_in_ref.file().relname():
				return False
		return True


class JavaUnderstandDatabase(UnderstandDatabase):
	def __init__(self, db, level, project_path, test_path):
		UnderstandDatabase.__init__(self, db, level, project_path, test_path)
		self.lang = "java"

	def get_ents(self):
		entity_kind = self.level
		if self.level == "function":
			entity_kind = "method"
		return self.db.ents(f"{self.lang} {entity_kind} ~unresolved ~unknown")

	def get_dependencies(self, entity):
		if self.level == "file":
			return list(entity.depends().keys())
		elif self.level == "function":
			return list(map(lambda ref: ref.ent(), entity.refs('java call')))

	def get_und_function_unique_name(self, und_function):
		und_parameters = und_function.parameters() if und_function.parameters() is not None else ""
		und_name = und_function.name().replace('.', '::') + f"({und_parameters})".replace(' ', '')
		return und_name

	def get_pydriller_function_unique_name(self, pydriller_function):
		return pydriller_function.long_name.replace(' ', '')

	def entity_is_valid(self, entity):
		if not UnderstandDatabase.entity_is_valid(self, entity):
			return False
		if self.level == "file" and ".class" in entity.name():
			return False
		if self.level == "function":
			if re.search("\(Anon_\d+\)", entity.name()) is not None:
				return False
			if re.search("\(lambda_expr_\d+\)", entity.name()) is not None:
				return False
		return True
