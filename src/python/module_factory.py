from .code_analyzer.code_analyzer import AnalysisLevel
from .entities.entity import Language
from .code_analyzer.understand.understand_analyzer import *
from .execution_record_extractor.travis_ci_extractor import *
from .execution_record_extractor.rtp_torrent_extractor import RTPTorrentExtractor
from .repository_miner import FileRepositoryMiner, FunctionRepositoryMiner


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
    def get_execution_record_extractor(language, rtp_path):
        if rtp_path is not None:
            return RTPTorrentExtractor
        extractor_class = None
        if language == Language.JAVA:
            extractor_class = TravisCIJavaExtractor
        elif language == Language.C:
            extractor_class = TravisCICExtractor
        return extractor_class

    @staticmethod
    def get_repository_miner(level):
        miner_class = None
        if level == AnalysisLevel.FILE:
            miner_class = FileRepositoryMiner
        elif level == AnalysisLevel.FUNCTION:
            miner_class = FunctionRepositoryMiner
        return miner_class