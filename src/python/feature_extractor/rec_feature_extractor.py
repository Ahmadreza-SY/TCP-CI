from ..entities.entity import Entity
from ..entities.entity_change import EntityChange
from ..entities.execution_record import ExecutionRecord, TestVerdict
from ..feature_extractor.feature import Feature


class RecFeatureExtractor:
    def __init__(self, build_window, exe_df, build_change_history):
        self.build_window = build_window
        self.exe_df = exe_df
        self.build_change_history = build_change_history

    def compute_rec_features(self, test_entities, build, build_tc_features):
        window = self.build_window
        builds = self.exe_df[ExecutionRecord.BUILD].unique().tolist()

        for test in test_entities.to_dict("records"):
            test_id = test[Entity.ID]
            build_tc_features.setdefault(test_id, {})

            test_result = self.exe_df[
                (self.exe_df[ExecutionRecord.BUILD] == build.id)
                & (self.exe_df[ExecutionRecord.TEST] == test_id)
            ].iloc[0]
            test_exe_df = (
                self.exe_df[self.exe_df[ExecutionRecord.TEST] == test_id]
                .copy()
                .reset_index(drop=True)
            )
            test_exe_history = (
                test_exe_df[
                    test_exe_df.index
                    < test_exe_df[test_exe_df[ExecutionRecord.BUILD] == build.id].index[
                        0
                    ]
                ]
                .copy()
                .reset_index(drop=True)
            )
            build_tc_features[test_id][Feature.VERDICT] = test_result[
                ExecutionRecord.VERDICT
            ]
            build_tc_features[test_id][Feature.DURATION] = test_result[
                ExecutionRecord.DURATION
            ]
            if test_exe_history.empty:
                for feature in Feature.rec_features:
                    build_tc_features[test_id][f"REC_{feature}"] = Feature.DEFAULT_VALUE
                build_tc_features[test_id][f"REC_{Feature.AGE}"] = 0
                continue

            test_exe_history["transition"] = (
                test_exe_history[ExecutionRecord.VERDICT].diff().fillna(0) != 0
            ).astype(int)
            test_exe_recent = test_exe_history.tail(window)

            first_build_id = test_exe_df.iloc[0][ExecutionRecord.BUILD]
            last_build_id = test_result[ExecutionRecord.BUILD]
            age = builds.index(last_build_id) - builds.index(first_build_id)
            last_failure_age = -1
            last_transition_age = -1
            if len(test_exe_history[test_exe_history[ExecutionRecord.VERDICT] > 0]) > 0:
                last_failure_age = (
                    test_exe_history.index.max()
                    - test_exe_history[
                        test_exe_history[ExecutionRecord.VERDICT] > 0
                    ].index[-1]
                )
            if len(test_exe_history[test_exe_history["transition"] > 0]) > 0:
                last_transition_age = (
                    test_exe_history.index.max()
                    - test_exe_history[test_exe_history["transition"] > 0].index[-1]
                )

            # Recent
            recent_avg_duration = test_exe_recent[ExecutionRecord.DURATION].mean()
            recent_max_duration = test_exe_recent[ExecutionRecord.DURATION].max()
            (
                recent_fail_rate,
                recent_assert_rate,
                recent_exc_rate,
                recent_transition_rate,
            ) = self.compute_test_rates(test_exe_recent)
            # Total
            total_avg_duration = test_exe_history[ExecutionRecord.DURATION].mean()
            total_max_duration = test_exe_history[ExecutionRecord.DURATION].max()
            (
                total_fail_rate,
                total_assert_rate,
                total_exc_rate,
                total_transition_rate,
            ) = self.compute_test_rates(test_exe_history)

            last_verdict = test_exe_recent.tail(1)[ExecutionRecord.VERDICT].values[0]
            last_duration = test_exe_recent.tail(1)[ExecutionRecord.DURATION].values[0]

            build_changed_ents = (
                self.build_change_history[
                    self.build_change_history["BuildId"] == build.id
                ][EntityChange.ID]
                .unique()
                .tolist()
            )
            max_test_file_fail_rate = self.compute_max_test_file_rate(
                test_exe_history, ExecutionRecord.VERDICT, build_changed_ents
            )
            max_test_file_transition_rate = self.compute_max_test_file_rate(
                test_exe_history, "transition", build_changed_ents
            )

            features = [
                # Age
                (f"REC_{Feature.AGE}", age),
                (f"REC_{Feature.LAST_FAILURE_AGE}", last_failure_age),
                (f"REC_{Feature.LAST_TRANSITION_AGE}", last_transition_age),
                # Recent
                (f"REC_{Feature.RECENT_AVG_EXE_TIME}", recent_avg_duration),
                (f"REC_{Feature.RECENT_MAX_EXE_TIME}", recent_max_duration),
                (f"REC_{Feature.RECENT_FAIL_RATE}", recent_fail_rate),
                (f"REC_{Feature.RECENT_ASSERT_RATE}", recent_assert_rate),
                (f"REC_{Feature.RECENT_EXC_RATE}", recent_exc_rate),
                (f"REC_{Feature.RECENT_TRANSITION_RATE}", recent_transition_rate),
                # Total
                (f"REC_{Feature.TOTAL_AVG_EXE_TIME}", total_avg_duration),
                (f"REC_{Feature.TOTAL_MAX_EXE_TIME}", total_max_duration),
                (f"REC_{Feature.TOTAL_FAIL_RATE}", total_fail_rate),
                (f"REC_{Feature.TOTAL_ASSERT_RATE}", total_assert_rate),
                (f"REC_{Feature.TOTAL_EXC_RATE}", total_exc_rate),
                (f"REC_{Feature.TOTAL_TRANSITION_RATE}", total_transition_rate),
                # Last
                (f"REC_{Feature.LAST_VERDICT}", last_verdict),
                (f"REC_{Feature.LAST_EXE_TIME}", last_duration),
                # Test File Rates
                (f"REC_{Feature.MAX_TEST_FILE_FAIL_RATE}", max_test_file_fail_rate),
                (
                    f"REC_{Feature.MAX_TEST_FILE_TRANSITION_RATE}",
                    max_test_file_transition_rate,
                ),
            ]

            for name, value in features:
                build_tc_features[test_id][name] = value

        return build_tc_features

    def compute_test_rates(self, test_exe_history):
        fail_rate = len(
            test_exe_history[
                test_exe_history[ExecutionRecord.VERDICT] != TestVerdict.SUCCESS.value
            ]
        ) / len(test_exe_history)
        assert_rate = len(
            test_exe_history[
                test_exe_history[ExecutionRecord.VERDICT] == TestVerdict.ASSERTION.value
            ]
        ) / len(test_exe_history)
        exc_rate = len(
            test_exe_history[
                test_exe_history[ExecutionRecord.VERDICT] == TestVerdict.EXCEPTION.value
            ]
        ) / len(test_exe_history)
        transition_rate = len(
            test_exe_history[test_exe_history["transition"] == 1]
        ) / len(test_exe_history)
        return fail_rate, assert_rate, exc_rate, transition_rate

    def compute_max_test_file_rate(
        self, test_exe_history, target_col, build_changed_ents
    ):
        max_test_file_rate = -1
        test_target_builds = (
            test_exe_history[test_exe_history[target_col] > 0][ExecutionRecord.BUILD]
            .unique()
            .tolist()
        )
        if len(test_target_builds) > 0:
            test_file_target_history = (
                self.build_change_history[
                    (
                        self.build_change_history[EntityChange.ID].isin(
                            build_changed_ents
                        )
                    )
                    & (self.build_change_history["BuildId"].isin(test_target_builds))
                ]
                .groupby([EntityChange.ID, "BuildId"], as_index=False)
                .first()
            )
            if len(test_file_target_history) == 0:
                max_test_file_rate = 0.0
            else:
                max_target_freq = (
                    test_file_target_history.groupby([EntityChange.ID], as_index=False)
                    .count()["BuildId"]
                    .max()
                )
                max_test_file_rate = max_target_freq / len(test_target_builds)
        return max_test_file_rate
