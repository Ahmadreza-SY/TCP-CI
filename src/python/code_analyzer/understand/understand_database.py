import os
import re
import glob
import sys
import time
import subprocess
import shlex
from tqdm import tqdm
import understand
from ...entities.entity import Language, EntityType
from ..code_analyzer import AnalysisLevel


class UnderstandDatabase:
    language_map = {Language.JAVA: "java", Language.C: "c++"}

    def __init__(
        self, project_path: str, test_path: str, output_path: str, level: AnalysisLevel
    ):
        self.db = None
        self.project_path = project_path
        self.test_path = test_path
        self.level = level
        self.output_path = output_path

    def get_und_db(self):
        project_name = self.project_path.split("/")[-1]
        und_db_path = f"{self.output_path}/{project_name}.udb"
        if not os.path.isfile(und_db_path):
            start = time.time()
            language_argument = UnderstandDatabase.language_map[self.language]
            print("Running understand analysis")
            und_command = f"und -verbose -db {und_db_path} create -languages {language_argument} add {self.project_path} analyze"
            self.run_und_command(und_command)
            print(
                f'Created understand db at {und_db_path}, took {"{0:.2f}".format(time.time() - start)} seconds.'
            )
        if self.db is None:
            print("Loading understand database ...")
            self.db = understand.open(und_db_path)
        return self.db

    def run_und_command(self, command):
        pbar = None
        full_project_path = os.path.abspath(self.project_path)
        process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
        while True:
            output = process.stdout.readline().decode("utf-8").strip()
            if output == "" and process.poll() is not None:
                break
            if output:
                if "Files added:" in output:
                    pbar_total = int(output.split(" ")[-1])
                    if self.language == Language.JAVA:
                        pbar_total *= 2
                    pbar = tqdm(
                        total=pbar_total, file=sys.stdout, desc="Analyzing files ..."
                    )
                elif (
                    "File:" not in output
                    and "Warning:" not in output
                    and pbar is not None
                ):
                    if "RELATIVE:/" in output or full_project_path in output:
                        pbar.update(1)
        if pbar is None:
            print("No output captured from understand!")
            sys.exit()
        pbar.close()
        rc = process.poll()
        return rc

    def entity_is_valid(self, entity):
        entity_file_path = None
        if self.level == AnalysisLevel.FILE:
            entity_file_path = self.get_valid_rel_path(entity)
        elif self.level == AnalysisLevel.FUNCTION:
            define_ref = entity.ref("definein")
            if define_ref is None:
                return False
            entity_file_path = self.get_valid_rel_path(define_ref.file())
        if entity_file_path is not None and entity_file_path[0] == "/":
            return False
        return True

    def get_valid_rel_path(self, entity):
        full_project_path = os.path.abspath(self.project_path)
        full_path = entity.longname()
        if full_project_path not in full_path and "RELATIVE:/" in full_path:
            full_path = full_path.replace("RELATIVE:/", "")
            full_path = "/".join(full_path.split("/")[1:])
        return full_path.replace(full_project_path + "/", "")

    def get_entity_type(self, entity, rel_path):
        if self.test_path is None:
            print("Test path cannot be None for this configuration. Aborting ...")
            sys.exit()
        file_name = rel_path.split("/")[-1]
        full_test_path = os.path.abspath(f"{self.project_path}/{self.test_path}")
        pattern = f"{full_test_path}/**/{file_name}"
        for match in glob.glob(pattern, recursive=True):
            if os.path.isfile(match) and rel_path in match:
                return EntityType.TEST
        return EntityType.SRC

    def get_ents_by_id(self, ids):
        return list(map(lambda id: self.get_und_db().ent_from_id(id), ids))

    def get_ents(self):
        pass

    def get_dependencies(self, entity):
        pass

    def get_und_function_unique_name(self, und_function):
        pass


class UnderstandCDatabase(UnderstandDatabase):
    def __init__(
        self, project_path: str, test_path: str, output_path: str, level: AnalysisLevel
    ):
        UnderstandDatabase.__init__(self, project_path, test_path, output_path, level)
        self.language = Language.C

    def get_ents(self):
        language_argument = UnderstandDatabase.language_map[self.language]
        level_argument = None
        if self.level == AnalysisLevel.FILE:
            level_argument = "file"
        elif self.level == AnalysisLevel.FUNCTION:
            level_argument = "function"
        return self.get_und_db().ents(
            f"{language_argument} {level_argument} ~unresolved ~unknown"
        )

    def get_dependencies(self, entity):
        if self.level == AnalysisLevel.FILE:
            return list(entity.depends().keys())
        elif self.level == AnalysisLevel.FUNCTION:
            return list(map(lambda ref: ref.ent(), entity.refs("c call")))

    def get_und_function_unique_name(self, und_function):
        return und_function.longname()

    def entity_is_valid(self, entity):
        if not UnderstandDatabase.entity_is_valid(self, entity):
            return False
        if self.level == AnalysisLevel.FILE and "/_deps/" in entity.relname():
            return False
        elif self.level == AnalysisLevel.FUNCTION:
            if entity.name() == "[unnamed]":
                return False
            define_in_ref = entity.ref("definein")
            if define_in_ref is None or "/_deps/" in define_in_ref.file().relname():
                return False
        return True


class UnderstandJavaDatabase(UnderstandDatabase):
    def __init__(
        self, project_path: str, test_path: str, output_path: str, level: AnalysisLevel
    ):
        UnderstandDatabase.__init__(self, project_path, test_path, output_path, level)
        self.language = Language.JAVA

    def get_entity_type(self, entity, rel_path):
        if self.test_path is not None:
            return super().get_entity_type(entity, rel_path)
        if "src/test" in rel_path:
            return EntityType.TEST
        else:
            return EntityType.SRC

    def get_ents(self):
        language_argument = UnderstandDatabase.language_map[self.language]
        level_argument = None
        if self.level == AnalysisLevel.FILE:
            level_argument = "file"
        elif self.level == AnalysisLevel.FUNCTION:
            level_argument = "method"
        return self.get_und_db().ents(
            f"{language_argument} {level_argument} ~unresolved ~unknown"
        )

    def get_dependencies(self, entity):
        if self.level == AnalysisLevel.FILE:
            return list(entity.depends().keys())
        elif self.level == AnalysisLevel.FUNCTION:
            return list(map(lambda ref: ref.ent(), entity.refs("java call")))

    def get_und_function_unique_name(self, und_function):
        und_parameters = (
            und_function.parameters() if und_function.parameters() is not None else ""
        )
        und_name = und_function.name().replace(
            ".", "::"
        ) + f"({und_parameters})".replace(" ", "")
        return und_name

    def entity_is_valid(self, entity):
        if not UnderstandDatabase.entity_is_valid(self, entity):
            return False
        if self.level == AnalysisLevel.FILE and ".class" in entity.name():
            return False
        if self.level == AnalysisLevel.FUNCTION:
            if re.search("\(Anon_\d+\)", entity.name()) is not None:
                return False
            if re.search("\(lambda_expr_\d+\)", entity.name()) is not None:
                return False
        return True
