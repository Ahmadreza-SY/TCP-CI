from .code_analyzer.code_analyzer import AnalysisLevel
from .entities.entity import Language
from .code_analyzer.understand.understand_analyzer import *
from .execution_record_extractor.travis_ci_extractor import *


class ModuleFactory:
    @staticmethod
    def get_code_analyzer(level):
        analyzer_class = None
        if level == AnalysisLevel.FILE:
            analyzer_class = UnderstandFileAnalyzer
        elif level == AnalysisLevel.FUNCTION:
            analyzer_class = UnderstandFunctionAnalyzer
        return analyzer_class

    @staticmethod
    def get_execution_record_extractor(language):
        extractor_class = None
        if language == Language.JAVA:
            extractor_class = TravisCIJavaExtractor
        elif language == Language.C:
            extractor_class = TravisCICExtractor
        return extractor_class