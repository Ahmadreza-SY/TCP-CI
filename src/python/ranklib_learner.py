from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import subprocess
from .feature_extractor.feature import Feature
from pathlib import Path
from tqdm import tqdm
import sys
import re
import os


class RankLibLearner:

    PRED_COLS = [
        "qid",
        "Q",
        "target",
        "verdict",
        "duration",
        "test",
        "build",
        "no.",
        "score",
        "indri",
    ]

    def __init__(self, config):
        self.config = config
        self.feature_id_map_path = config.output_path / "feature_id_map.csv"
        if self.feature_id_map_path.exists():
            feature_id_map_df = pd.read_csv(self.feature_id_map_path)
            keys = feature_id_map_df["key"].values.tolist()
            values = feature_id_map_df["value"].values.tolist()
            self.feature_id_map = dict(zip(keys, values))
            self.next_fid = max(values) + 1
        else:
            self.feature_id_map = {}
            self.next_fid = 1
        builds_df = pd.read_csv(
            config.output_path / "builds.csv", parse_dates=["started_at"]
        )
        self.build_time_d = dict(
            zip(
                builds_df["id"].values.tolist(), builds_df["started_at"].values.tolist()
            )
        )

    def get_feature_id(self, feature_name):
        if feature_name not in self.feature_id_map:
            self.feature_id_map[feature_name] = self.next_fid
            self.next_fid += 1
        return self.feature_id_map[feature_name]

    def save_feature_id_map(self):
        keys = list(self.feature_id_map.keys())
        values = list(self.feature_id_map.values())
        feature_id_map_df = pd.DataFrame({"key": keys, "value": values})
        feature_id_map_df.to_csv(self.feature_id_map_path, index=False)

    def normalize_dataset(self, dataset, scaler):
        non_feature_cols = [
            Feature.BUILD,
            Feature.TEST,
            Feature.VERDICT,
            Feature.DURATION,
        ]
        feature_dataset = dataset.drop(non_feature_cols, axis=1)
        if scaler == None:
            scaler = MinMaxScaler()
            scaler.fit(feature_dataset)
        normalized_dataset = pd.DataFrame(
            scaler.transform(feature_dataset),
            columns=feature_dataset.columns,
        )
        for col in non_feature_cols:
            normalized_dataset[col] = dataset[col]

        return normalized_dataset, feature_dataset, scaler

    def convert_to_ranklib_dataset(self, dataset, scaler=None):
        if dataset.empty:
            return None
        dataset = dataset.copy()
        dataset[Feature.VERDICT] = dataset[Feature.VERDICT].apply(lambda v: int(v > 0))
        normalized_dataset, feature_dataset, _ = self.normalize_dataset(dataset, scaler)
        builds = normalized_dataset[Feature.BUILD].unique()
        ranklib_ds_rows = []
        for i, build in list(enumerate(builds)):
            build_ds = normalized_dataset[
                normalized_dataset[Feature.BUILD] == build
            ].copy()
            build_ds["B_Verdict"] = (build_ds[Feature.VERDICT] > 0).astype(int)
            build_ds.sort_values(
                ["B_Verdict", Feature.DURATION],
                ascending=[False, True],
                inplace=True,
                ignore_index=True,
            )
            build_ds.drop("B_Verdict", axis=1, inplace=True)
            build_ds["Target"] = -build_ds.index + len(build_ds)
            for _, record in build_ds.iterrows():
                row_items = [int(record["Target"]), f"qid:{i+1}"]
                row_feature_items = []
                for _, f in enumerate(feature_dataset.columns):
                    fid = self.get_feature_id(f)
                    row_feature_items.append(f"{fid}:{record[f]}")
                row_feature_items.sort(key=lambda v: int(v.split(":")[0]))
                row_items.extend(row_feature_items)
                row_items.extend(
                    [
                        "#",
                        int(record["Target"]),
                        int(record[Feature.VERDICT]),
                        int(record[Feature.DURATION]),
                        int(record[Feature.TEST]),
                        int(record[Feature.BUILD]),
                    ]
                )
                ranklib_ds_rows.append(row_items)
        headers = (
            ["target", "qid"]
            + [f"f{i+1}" for i in range(len(feature_dataset.columns))]
            + ["hashtag", "i_target", "i_verdict", "i_duration", "i_test", "i_build"]
        )
        self.save_feature_id_map()
        return pd.DataFrame(ranklib_ds_rows, columns=headers)

    def create_ranklib_training_sets(self, ranklib_ds, output_path):
        builds = ranklib_ds["i_build"].unique().tolist()
        builds.sort(key=lambda b: self.build_time_d[b])
        test_builds = set(builds[-self.config.test_count :])
        for i, build in tqdm(list(enumerate(builds)), desc="Creating training sets"):
            if build not in test_builds:
                continue
            train_ds = ranklib_ds[ranklib_ds["i_build"].isin(builds[:i])]
            if len(train_ds) == 0:
                continue
            test_ds = ranklib_ds[ranklib_ds["i_build"] == build]
            build_out_path = output_path / str(build)
            build_out_path.mkdir(parents=True, exist_ok=True)
            if (
                not (output_path / str(build) / "train.txt").exists()
                and not (output_path / str(build) / "model.txt").exists()
            ):
                train_ds.to_csv(
                    output_path / str(build) / "train.txt",
                    sep=" ",
                    header=False,
                    index=False,
                )
            if not (output_path / str(build) / "test.txt").exists():
                test_ds.to_csv(
                    output_path / str(build) / "test.txt",
                    sep=" ",
                    header=False,
                    index=False,
                )

    def compute_apfd(self, pred):
        if len(set(pred["score"].values.tolist())) == 1 and len(pred) > 1:
            return 0.5
        n = len(pred)
        if n <= 1:
            return 1.0
        m = len(pred[pred["verdict"] > 0])
        fault_pos_sum = np.sum(pred[pred["verdict"] > 0].index + 1)
        apfd = 1 - fault_pos_sum / (n * m) + 1 / (2 * n)
        return float("{:.3f}".format(apfd))

    def compute_apfdc(self, pred):
        if len(set(pred["score"].values.tolist())) == 1 and len(pred) > 1:
            return 0.5
        n = len(pred)
        if n <= 1:
            return 1.0
        m = len(pred[pred["verdict"] > 0])
        costs = pred["duration"].values.tolist()
        failed_costs = 0.0
        for tfi in pred[pred["verdict"] > 0].index:
            failed_costs += sum(costs[tfi:]) - (costs[tfi] / 2)
        apfdc = failed_costs / (sum(costs) * m)
        return float("{:.3f}".format(apfdc))

    def extract_and_save_feature_stats(self, feature_stats_output, output_path):
        feature_freq_map = {i: 0 for i in self.feature_id_map.values()}
        matches = re.finditer(
            r"Feature\[(\d+)\]\s+:\s+(\d+)", feature_stats_output, re.MULTILINE
        )
        for match in matches:
            feature_id = int(match.group(1))
            feature_freq = int(match.group(2))
            feature_freq_map[feature_id] = feature_freq
        feature_stats_df = pd.DataFrame(
            {
                "feature_id": list(feature_freq_map.keys()),
                "frequency": list(feature_freq_map.values()),
            }
        )
        feature_stats_df.sort_values("feature_id", ignore_index=True, inplace=True)
        feature_stats_df.to_csv(output_path / "feature_stats.csv", index=False)

    def train_and_test(self, output_path, ranker):
        ranklib_path = Path("assets") / "RankLib.jar"
        math3_path = Path("assets") / "commons-math3.jar"
        results = {"build": [], "apfd": [], "apfdc": []}
        ds_paths = list(p for p in output_path.glob("*") if p.is_dir())
        for build_ds_path in tqdm(ds_paths, desc="Running full feature set training"):
            train_path = build_ds_path / "train.txt"
            test_path = build_ds_path / "test.txt"
            model_path = build_ds_path / "model.txt"
            pred_path = build_ds_path / "pred.txt"

            if not model_path.exists():
                train_command = f"java -jar {ranklib_path} -train {train_path} -ranker {ranker[0]} {ranker[1]} -save {model_path} -silent"
                train_out = subprocess.run(
                    train_command, shell=True, capture_output=True
                )
                if train_out.returncode != 0:
                    print(f"Error in training:\n{train_out.stderr}")
                    sys.exit()
                os.remove(str(train_path))
            if not (build_ds_path / "feature_stats.csv").exists():
                feature_stats_command = f"java -cp {ranklib_path}:{math3_path} ciir.umass.edu.features.FeatureManager -feature_stats {model_path}"
                feature_stats_out = subprocess.run(
                    feature_stats_command, shell=True, capture_output=True
                )
                if feature_stats_out.returncode != 0:
                    print(f"Error in training:\n{feature_stats_out.stderr}")
                    sys.exit()
                self.extract_and_save_feature_stats(
                    feature_stats_out.stdout.decode("utf-8"), build_ds_path
                )
            if not pred_path.exists():
                pred_command = f"java -jar {ranklib_path} -load {model_path} -rank {test_path} -indri {pred_path}"
                pred_out = subprocess.run(pred_command, shell=True, capture_output=True)
                if pred_out.returncode != 0:
                    print(f"Error in predicting:\n{pred_out.stderr}")
                    sys.exit()
            pred_df = (
                pd.read_csv(
                    pred_path,
                    sep=" ",
                    names=RankLibLearner.PRED_COLS,
                )
                # Shuffle predictions when predicted scores are equal to randomize the order.
                .sample(frac=1).reset_index(drop=True)
            )
            pred_df.sort_values(
                "score", ascending=False, inplace=True, ignore_index=True
            )
            apfd = self.compute_apfd(pred_df)
            apfdc = self.compute_apfdc(pred_df)
            results["build"].append(int(build_ds_path.name))
            results["apfd"].append(apfd)
            results["apfdc"].append(apfdc)
        results_df = pd.DataFrame(results)
        results_df["build_time"] = results_df["build"].apply(
            lambda b: self.build_time_d[b]
        )
        results_df.sort_values("build_time", ignore_index=True, inplace=True)
        results_df.drop("build_time", axis=1, inplace=True)
        return results_df

    def evaluate_heuristic(self, hname, suite_ds):
        asc_suite = suite_ds.sort_values(hname, ascending=True, ignore_index=True)
        asc_pred = pd.DataFrame(
            {
                "verdict": asc_suite[Feature.VERDICT].values,
                "duration": asc_suite[Feature.DURATION].values,
                "score": asc_suite[hname].values,
            }
        )
        apfd_asc = self.compute_apfd(asc_pred)
        apfdc_asc = self.compute_apfdc(asc_pred)

        dsc_suite = suite_ds.sort_values(hname, ascending=False, ignore_index=True)
        dsc_pred = pd.DataFrame(
            {
                "verdict": dsc_suite[Feature.VERDICT].values,
                "duration": dsc_suite[Feature.DURATION].values,
                "score": dsc_suite[hname].values,
            }
        )
        apfd_dsc = self.compute_apfd(dsc_pred)
        apfdc_dsc = self.compute_apfdc(dsc_pred)

        return apfd_asc, apfd_dsc, apfdc_asc, apfdc_dsc

    def test_heuristics(self, dataset_df, results_path):
        apfd_results = {"build": []}
        apfdc_results = {"build": []}
        all_builds = dataset_df[Feature.BUILD].unique().tolist()
        all_builds.sort(key=lambda b: self.build_time_d[b])
        for build in tqdm(all_builds, desc="Testing heuristics"):
            suite_ds = dataset_df[dataset_df[Feature.BUILD] == build]
            apfd_results["build"].append(build)
            apfdc_results["build"].append(build)
            for fname, fid in self.feature_id_map.items():
                apfd_asc, apfd_dsc, apfdc_asc, apfdc_dsc = self.evaluate_heuristic(
                    fname, suite_ds
                )
                apfd_results.setdefault(f"{fid}-asc", []).append(apfd_asc)
                apfd_results.setdefault(f"{fid}-dsc", []).append(apfd_dsc)
                apfdc_results.setdefault(f"{fid}-asc", []).append(apfdc_asc)
                apfdc_results.setdefault(f"{fid}-dsc", []).append(apfdc_dsc)
        pd.DataFrame(apfd_results).to_csv(
            results_path / "heuristic_apfd_results.csv", index=False
        )
        pd.DataFrame(apfdc_results).to_csv(
            results_path / "heuristic_apfdc_results.csv", index=False
        )

    def run_accuracy_experiments(self, dataset_df, name, results_path, ranker=(0, "")):
        ranklib_ds = self.convert_to_ranklib_dataset(dataset_df)
        traning_sets_path = results_path / name
        self.create_ranklib_training_sets(ranklib_ds, traning_sets_path)
        results = self.train_and_test(traning_sets_path, ranker)
        results.to_csv(traning_sets_path / "results.csv", index=False)

    def convert_decay_datasets(self, datasets_path):
        original_ds_df = pd.read_csv(self.config.output_path / "dataset.csv")
        _, _, scaler = self.normalize_dataset(original_ds_df, None)
        decay_dataset_paths = list(p for p in datasets_path.glob("*") if p.is_dir())
        for decay_dataset_path in tqdm(
            decay_dataset_paths, desc="Converting decay datasets"
        ):
            test_file = decay_dataset_path / "test.txt"
            if test_file.exists():
                continue

            decay_ds_df = pd.read_csv(decay_dataset_path / "dataset.csv")
            # Reoder columns for MinMaxScaler
            decay_ds_df = decay_ds_df[original_ds_df.columns.tolist()]

            decay_ranklib_ds = self.convert_to_ranklib_dataset(decay_ds_df, scaler)
            decay_ranklib_ds.to_csv(
                test_file,
                sep=" ",
                header=False,
                index=False,
            )

    def test_decay_datasets(self, eval_model_paths, datasets_path):
        ranklib_path = Path("assets") / "RankLib.jar"
        for model_path in tqdm(eval_model_paths, desc=f"Testing models"):
            model_file = model_path / "model.txt"
            test_file = datasets_path / model_path.name / "test.txt"
            pred_file = datasets_path / model_path.name / "pred.txt"
            pred_command = f"java -jar {ranklib_path} -load {model_file} -rank {test_file} -indri {pred_file}"
            pred_out = subprocess.run(pred_command, shell=True, capture_output=True)
            if pred_out.returncode != 0:
                print(f"Error in predicting:\n{pred_out.stderr}")
                sys.exit()
            preds_df = (
                pd.read_csv(
                    pred_file,
                    sep=" ",
                    names=RankLibLearner.PRED_COLS,
                )
                # Shuffle predictions when predicted scores are equal to randomize the order.
                .sample(frac=1).reset_index(drop=True)
            )
            pred_builds = preds_df["build"].unique().tolist()
            results = {"build": [], "apfd": [], "apfdc": []}
            for pred_build in pred_builds:
                pred_df = (
                    preds_df[preds_df["build"] == pred_build]
                    .copy()
                    .reset_index(drop=True)
                    .sort_values("score", ascending=False, ignore_index=True)
                )
                apfd = self.compute_apfd(pred_df)
                apfdc = self.compute_apfdc(pred_df)
                results["build"].append(pred_build)
                results["apfd"].append(apfd)
                results["apfdc"].append(apfdc)
            results_df = pd.DataFrame(results)
            results_df["build_time"] = results_df["build"].apply(
                lambda b: self.build_time_d[b]
            )
            results_df.sort_values("build_time", ignore_index=True, inplace=True)
            results_df.drop("build_time", axis=1, inplace=True)
            results_df.to_csv(
                datasets_path / model_path.name / "results.csv",
                index=False,
            )

    def run_decay_test_experiments(self, datasets_path, models_path):
        self.convert_decay_datasets(datasets_path)
        model_paths = list(p for p in models_path.glob("*") if p.is_dir())
        self.test_decay_datasets(model_paths, datasets_path)
