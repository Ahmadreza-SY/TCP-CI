from .code_analyzer.code_analyzer import AnalysisLevel
from .code_analyzer.understand.understand_analyzer import *
from .execution_record_extractor.rtp_torrent_extractor import TorrentExtractor
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
    def get_execution_record_extractor(language, ci_data_path):
        return TorrentExtractor

    @staticmethod
    def get_repository_miner(level):
        miner_class = None
        if level == AnalysisLevel.FILE:
            miner_class = FileRepositoryMiner
        elif level == AnalysisLevel.FUNCTION:
            miner_class = FunctionRepositoryMiner
        return miner_class