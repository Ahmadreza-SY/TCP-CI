from .entities.entity import Entity
from .entities.entity_change import EntityChange
from .entities.execution_record import ExecutionRecord, Build
from tqdm import tqdm
import pandas as pd
from .feature_extractor.feature import Feature
from .feature_extractor.rec_feature_extractor import RecFeatureExtractor


class DecayDatasetFactory:
    def __init__(self, ds_factory, config):
        self.ds_factory = ds_factory
        self.config = config

    def update_test_file_features(
        self, features, rec_extractor, test_build_changed_ents, og_build, test_exe_df
    ):
        test_exe_history = (
            test_exe_df[
                test_exe_df.index
                < test_exe_df[test_exe_df[ExecutionRecord.BUILD] == og_build.id].index[
                    0
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )
        test_exe_history["transition"] = (
            test_exe_history[ExecutionRecord.VERDICT].diff().fillna(0) != 0
        ).astype(int)
        features[
            Feature.MAX_TEST_FILE_FAIL_RATE
        ] = rec_extractor.compute_max_test_file_rate(
            test_exe_history, ExecutionRecord.VERDICT, test_build_changed_ents
        )
        features[
            Feature.MAX_TEST_FILE_TRANSITION_RATE
        ] = rec_extractor.compute_max_test_file_rate(
            test_exe_history, "transition", test_build_changed_ents
        )

    def create_decay_dataset(
        self, dataset_df, og_build, test_builds, all_builds, exe_df
    ):
        og_build_dataset = dataset_df[dataset_df[Feature.BUILD] == og_build.id].copy()
        og_build_tc_features = {}
        for _, row in og_build_dataset.iterrows():
            test_id = row[Feature.TEST]
            og_build_tc_features[test_id] = {}
            for f in Feature.TES_COM + Feature.TES_PRO:
                og_build_tc_features[test_id][f] = row[f]
        og_test_ids = og_build_dataset[Feature.TEST].values.tolist()
        og_metadata_path = (
            self.ds_factory.repository_miner.get_analysis_path(og_build)
            / "metadata.csv"
        )
        og_entities_df = pd.read_csv(og_metadata_path)
        og_entities_dict = {e[Entity.ID]: e for e in og_entities_df.to_dict("records")}
        og_src_ids = list(
            set(og_entities_df[Entity.ID].values.tolist()) - set(og_test_ids)
        )

        build_change_history = self.ds_factory.create_build_change_history(all_builds)
        rec_extractor = RecFeatureExtractor(
            self.config.build_window, exe_df, build_change_history
        )

        decay_dataset = []
        for test_build in test_builds:
            test_build_dataset = dataset_df[
                dataset_df[Feature.BUILD] == test_build.id
            ].copy()
            test_ids = test_build_dataset[Feature.TEST].values.tolist()
            og_coverage = self.ds_factory.compute_test_coverage(
                og_build,
                og_test_ids,
                og_src_ids,
                test_build,
            )

            test_build_tc_features = {}
            self.ds_factory.compute_cov_features(
                test_build, test_ids, og_coverage, test_build_tc_features
            )
            self.ds_factory.compute_cod_cov_features(
                og_build,
                test_ids,
                og_coverage,
                og_entities_dict,
                test_build_tc_features,
                test_build,
            )
            self.ds_factory.compute_det_features(
                og_build, test_ids, og_coverage, test_build_tc_features
            )
            test_build_changed_ents = (
                build_change_history[build_change_history["BuildId"] == test_build.id][
                    EntityChange.ID
                ]
                .unique()
                .tolist()
            )

            for _, row in test_build_dataset.iterrows():
                features = {}
                test_id = row[Feature.TEST]
                features[Feature.BUILD] = test_build.id
                features[Feature.TEST] = test_id
                features[Feature.VERDICT] = row[Feature.VERDICT]
                features[Feature.DURATION] = row[Feature.DURATION]
                for f in Feature.TES_COM + Feature.TES_PRO:
                    if test_id not in og_build_tc_features:
                        features[f] = None
                        continue
                    features[f] = og_build_tc_features[test_id][f]
                for f in Feature.TES_CHN + Feature.REC:
                    features[f] = row[f]

                test_exe_df = (
                    exe_df[exe_df[ExecutionRecord.TEST] == test_id]
                    .copy()
                    .reset_index(drop=True)
                )
                self.update_test_file_features(
                    features,
                    rec_extractor,
                    test_build_changed_ents,
                    og_build,
                    test_exe_df,
                )

                for f in (
                    Feature.COV
                    + Feature.COD_COV_COM
                    + Feature.COD_COV_PRO
                    + Feature.COD_COV_CHN
                    + Feature.DET_COV
                ):
                    if test_id not in test_build_tc_features:
                        features[f] = None
                        continue
                    features[f] = test_build_tc_features[test_id][f]
                decay_dataset.append(features)

        return decay_dataset

    def create_decay_datasets(self, dataset_df):
        all_builds_df = pd.read_csv(
            self.config.output_path / "builds.csv",
            parse_dates=["started_at"],
        )
        all_builds = all_builds_df["id"].values.tolist()
        build_to_commits = dict(
            zip(
                all_builds_df["id"].values.tolist(),
                [c.split("#") for c in all_builds_df["commits"].values],
            )
        )
        build_to_time = dict(
            zip(
                all_builds_df["id"].values.tolist(),
                all_builds_df["started_at"].values.tolist(),
            )
        )
        builds = dataset_df[Feature.BUILD].unique().tolist()
        builds.sort(key=lambda b: build_to_time[b])

        exe_df = pd.read_csv(self.config.output_path / "exe.csv")
        exe_df["started_at"] = exe_df[ExecutionRecord.BUILD].apply(
            lambda b: build_to_time[b]
        )
        exe_df.sort_values("started_at", ignore_index=True, inplace=True)
        exe_df.drop("started_at", inplace=True, axis=1)

        decay_datasets_path = self.config.output_path / "decay_datasets"
        decay_datasets_path.mkdir(parents=True, exist_ok=True)
        for i, build_id in tqdm(
            enumerate(builds), desc="Creating decay datasets", total=len(builds)
        ):
            if i == 0:
                continue

            dataset_path = decay_datasets_path / str(build_id)
            if (dataset_path / "dataset.csv").exists():
                continue

            og_build = Build(
                build_id, build_to_commits[build_id], build_to_time[build_id]
            )
            test_builds = [
                Build(
                    test_build_id,
                    build_to_commits[test_build_id],
                    build_to_time[test_build_id],
                )
                for test_build_id in builds[i:]
            ]
            decay_dataset = self.create_decay_dataset(
                dataset_df, og_build, test_builds, all_builds, exe_df
            )
            decay_dataset_df = pd.DataFrame.from_records(decay_dataset)
            decay_dataset_df = decay_dataset_df.fillna(decay_dataset_df.mean())
            dataset_path.mkdir(parents=True, exist_ok=True)
            decay_dataset_df.to_csv(dataset_path / "dataset.csv", index=False)