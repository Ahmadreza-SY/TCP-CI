class Feature:
    DEFAULT_VALUE = -1
    TEST = "Test"
    BUILD = "Build"
    # Complexity Metrics
    complexity_metrics = [
        "CountDeclFunction",
        "CountLine",
        "CountLineBlank",
        "CountLineCode",
        "CountLineCodeDecl",
        "CountLineCodeExe",
        "CountLineComment",
        "CountStmt",
        "CountStmtDecl",
        "CountStmtExe",
        "RatioCommentToCode",
        "MaxCyclomatic",
        "MaxCyclomaticModified",
        "MaxCyclomaticStrict",
        "MaxEssential",
        "MaxNesting",
        "SumCyclomatic",
        "SumCyclomaticModified",
        "SumCyclomaticStrict",
        "SumEssential",
        "CountDeclClass",
        "CountDeclClassMethod",
        "CountDeclClassVariable",
        "CountDeclExecutableUnit",
        "CountDeclInstanceMethod",
        "CountDeclInstanceVariable",
        "CountDeclMethod",
        "CountDeclMethodDefault",
        "CountDeclMethodPrivate",
        "CountDeclMethodProtected",
        "CountDeclMethodPublic",
    ]
    # Process Metrics
    COMMIT_COUNT = "CommitCount"
    D_DEV_COUNT = "DistinctDevCount"
    OWNERS_CONTRIBUTION = "OwnersContribution"
    MINOR_CONTRIBUTOR_COUNT = "MinorContributorCount"
    OWNERS_EXPERIENCE = "OwnersExperience"
    ALL_COMMITERS_EXPERIENCE = "AllCommitersExperience"
    process_metrics = [
        COMMIT_COUNT,
        D_DEV_COUNT,
        OWNERS_CONTRIBUTION,
        MINOR_CONTRIBUTOR_COUNT,
        OWNERS_EXPERIENCE,
        ALL_COMMITERS_EXPERIENCE,
    ]
    # Change Metrics
    LINES_ADDED = "LinesAdded"
    LINES_DELETED = "LinesDeleted"
    ADDED_CHANGE_SCATTERING = "AddedChangeScattering"
    DELETED_CHANGE_SCATTERING = "DeletedChangeScattering"
    DMM_SIZE = "DMMSize"
    DMM_COMPLEXITY = "DMMComplexity"
    DMM_INTERFACING = "DMMInterfacing"
    change_metrics = [
        LINES_ADDED,
        LINES_DELETED,
        ADDED_CHANGE_SCATTERING,
        DELETED_CHANGE_SCATTERING,
        DMM_SIZE,
        DMM_COMPLEXITY,
        DMM_INTERFACING,
    ]
    # All Metrics
    all_metrics = complexity_metrics + process_metrics + change_metrics
    # REC Features
    AGE = "Age"
    LAST_FAILURE_AGE = "LastFailureAge"
    LAST_TRANSITION_AGE = "LastTransitionAge"
    RECENT_AVG_EXE_TIME = "RecentAvgExeTime"
    RECENT_MAX_EXE_TIME = "RecentMaxExeTime"
    RECENT_FAIL_RATE = "RecentFailRate"
    RECENT_ASSERT_RATE = "RecentAssertRate"
    RECENT_EXC_RATE = "RecentExcRate"
    RECENT_TRANSITION_RATE = "RecentTransitionRate"
    TOTAL_AVG_EXE_TIME = "TotalAvgExeTime"
    TOTAL_MAX_EXE_TIME = "TotalMaxExeTime"
    TOTAL_FAIL_RATE = "TotalFailRate"
    TOTAL_ASSERT_RATE = "TotalAssertRate"
    TOTAL_EXC_RATE = "TotalExcRate"
    TOTAL_TRANSITION_RATE = "TotalTransitionRate"
    LAST_VERDICT = "LastVerdict"
    LAST_EXE_TIME = "LastExeTime"
    MAX_TEST_FILE_FAIL_RATE = "MaxTestFileFailRate"
    MAX_TEST_FILE_TRANSITION_RATE = "MaxTestFileTransitionRate"
    rec_features = [
        AGE,
        LAST_FAILURE_AGE,
        LAST_TRANSITION_AGE,
        RECENT_AVG_EXE_TIME,
        RECENT_MAX_EXE_TIME,
        RECENT_FAIL_RATE,
        RECENT_ASSERT_RATE,
        RECENT_EXC_RATE,
        RECENT_TRANSITION_RATE,
        TOTAL_AVG_EXE_TIME,
        TOTAL_MAX_EXE_TIME,
        TOTAL_FAIL_RATE,
        TOTAL_ASSERT_RATE,
        TOTAL_EXC_RATE,
        TOTAL_TRANSITION_RATE,
        LAST_VERDICT,
        LAST_EXE_TIME,
        MAX_TEST_FILE_FAIL_RATE,
        MAX_TEST_FILE_TRANSITION_RATE,
    ]
    # COV Features
    CHN_SCORE_SUM = "ChnScoreSum"
    IMP_SCORE_SUM = "ImpScoreSum"
    CHN_COUNT = "ChnCount"
    IMP_COUNT = "ImpCount"
    cov_features = [CHN_SCORE_SUM, IMP_SCORE_SUM, CHN_COUNT, IMP_COUNT]
    cov_impacted = [IMP_SCORE_SUM, IMP_COUNT]
    # DET Features
    FAULTS = "Faults"
    det_features = [FAULTS]
    # Ground Truth
    VERDICT = "Verdict"
    DURATION = "Duration"

    # Feature Groups' feature names
    TES_COM = [f"TES_COM_{m}" for m in complexity_metrics]
    TES_PRO = [f"TES_PRO_{m}" for m in process_metrics]
    TES_CHN = [f"TES_CHN_{m}" for m in change_metrics]
    REC = [f"REC_{m}" for m in rec_features]
    COV = [f"COV_{m}" for m in cov_features]
    COD_COV_COM = [f"COD_COV_COM_C_{m}" for m in complexity_metrics] + [
        f"COD_COV_COM_IMP_{m}" for m in complexity_metrics
    ]
    COD_COV_PRO = [f"COD_COV_PRO_C_{m}" for m in process_metrics] + [
        f"COD_COV_PRO_IMP_{m}" for m in process_metrics
    ]
    COD_COV_CHN = [f"COD_COV_CHN_C_{m}" for m in change_metrics]
    DET_COV = [f"DET_COV_C_{m}" for m in det_features] + [
        f"DET_COV_IMP_{m}" for m in det_features
    ]
    # Features' names which are related to impacted files
    IMPACTED_FEATURES = [f for f in COV if "_Imp" in f] + [
        f for f in COD_COV_COM + COD_COV_PRO + DET_COV if "_IMP_" in f
    ]

    @staticmethod
    def get_metric_prefix(metric):
        prefix = None
        if metric in Feature.complexity_metrics:
            prefix = "COM"
        elif metric in Feature.process_metrics:
            prefix = "PRO"
        elif metric in Feature.change_metrics:
            prefix = "CHN"
        return prefix