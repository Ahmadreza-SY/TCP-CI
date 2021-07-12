from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import subprocess
from .dataset_factory import DatasetFactory
from pathlib import Path
from tqdm import tqdm
import sys
import re


class RankLibLearner:

    PRED_COLS = [
        "qid",
        "Q",
        "target",
        "verdict",
        "test",
        "build",
        "no.",
        "score",
        "indri",
    ]

    def __init__(self, config):
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

    def convert_to_ranklib_dataset(self, dataset):
        if dataset.empty:
            return None
        dataset = dataset.copy()
        dataset[DatasetFactory.VERDICT] = dataset[DatasetFactory.VERDICT].apply(
            lambda v: int(v > 0)
        )
        non_feature_cols = [
            DatasetFactory.BUILD,
            DatasetFactory.TEST,
            DatasetFactory.VERDICT,
            DatasetFactory.DURATION,
        ]
        feature_dataset = dataset.drop(non_feature_cols, axis=1)
        normalized_dataset = pd.DataFrame(
            MinMaxScaler().fit_transform(feature_dataset),
            columns=feature_dataset.columns,
        )
        for col in non_feature_cols:
            normalized_dataset[col] = dataset[col]
        builds = normalized_dataset[DatasetFactory.BUILD].unique()
        ranklib_ds_rows = []
        for i, build in list(enumerate(builds)):
            build_ds = normalized_dataset[
                normalized_dataset[DatasetFactory.BUILD] == build
            ].copy()
            build_ds["B_Verdict"] = (build_ds[DatasetFactory.VERDICT] > 0).astype(int)
            build_ds.sort_values(
                ["B_Verdict", DatasetFactory.DURATION],
                ascending=[False, True],
                inplace=True,
                ignore_index=True,
            )
            build_ds.drop("B_Verdict", axis=1, inplace=True)
            build_ds["Target"] = -build_ds.index + len(build_ds)
            for _, record in build_ds.iterrows():
                row_items = [int(record["Target"]), f"qid:{i+1}"]
                for _, f in enumerate(feature_dataset.columns):
                    fid = self.get_feature_id(f)
                    row_items.append(f"{fid}:{record[f]}")
                row_items.extend(
                    [
                        "#",
                        int(record["Target"]),
                        int(record[DatasetFactory.VERDICT]),
                        int(record[DatasetFactory.TEST]),
                        int(record[DatasetFactory.BUILD]),
                    ]
                )
                ranklib_ds_rows.append(row_items)
        headers = (
            ["target", "qid"]
            + [f"f{i+1}" for i in range(len(feature_dataset.columns))]
            + ["hashtag", "i_target", "i_verdict", "i_test", "i_build"]
        )
        self.save_feature_id_map()
        return pd.DataFrame(ranklib_ds_rows, columns=headers)

    def create_ranklib_training_sets(self, ranklib_ds, output_path):
        builds = ranklib_ds["i_build"].unique()
        builds = np.sort(builds)
        for i, build in tqdm(list(enumerate(builds)), desc="Creating training sets"):
            if i == 0:
                continue
            train_ds = ranklib_ds[ranklib_ds["i_build"].isin(builds[:i])]
            test_ds = ranklib_ds[ranklib_ds["i_build"] == build]
            build_out_path = output_path / str(build)
            build_out_path.mkdir(parents=True, exist_ok=True)
            if not (output_path / str(build) / "train.txt").exists():
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

    def compute_napfd(self, pred):
        n = len(pred)
        m = len(pred[pred["verdict"] > 0])
        if n <= 1:
            return 1.0
        fault_pos_sum = np.sum(pred[pred["verdict"] > 0].index + 1)
        apfd = 1 - fault_pos_sum / (n * m) + 1 / (2 * n)
        napfd = (2 * n * apfd - m) / (2 * n - 2 * m)
        return float("{:.3f}".format(napfd))

    def compute_rpa(self, l):
        k = len(l)
        s = 0
        for i in range(k):
            s += sum(l[: (i + 1)])
        return (2 * s) / ((k ** 2) * (k + 1))

    def compute_nrpa(self, l):
        k = len(l)
        if k <= 1:
            return 1.0
        min_rpa = self.compute_rpa([i + 1 for i in range(k)])
        max_rpa = self.compute_rpa([i + 1 for i in reversed(range(k))])
        rpa = self.compute_rpa(l)
        nrpa = (rpa - min_rpa) / (max_rpa - min_rpa)
        return float("{:.3f}".format(nrpa))

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

    def train_and_test(self, eval_metric, output_path):
        ranklib_path = Path("assets") / "RankLib.jar"
        math3_path = Path("assets") / "commons-math3.jar"
        results = {"build": [], eval_metric: []}
        ds_paths = list(p for p in output_path.glob("*") if p.is_dir())
        for build_ds_path in tqdm(ds_paths, desc="Running full feature set training"):
            train_path = build_ds_path / "train.txt"
            test_path = build_ds_path / "test.txt"
            model_path = build_ds_path / "model.txt"
            pred_path = build_ds_path / "pred.txt"

            if not model_path.exists():
                train_command = f"java -jar {ranklib_path} -train {train_path} -ranker 0 -save {model_path} -metric2t NDCG@10 -silent"
                train_out = subprocess.run(
                    train_command, shell=True, capture_output=True
                )
                if train_out.returncode != 0:
                    print(f"Error in training:\n{train_out.stderr}")
                    sys.exit()
                train_path.unlink()
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
            eval_score = 0.0
            if eval_metric == "nrpa":
                eval_score = self.compute_nrpa(pred_df["target"].values.tolist())
            elif eval_metric == "napfd":
                eval_score = self.compute_napfd(pred_df)
            if len(set(pred_df["score"].values.tolist())) == 1:
                eval_score = 0.5
            results["build"].append(int(build_ds_path.name))
            results[eval_metric].append(eval_score)
        results_df = pd.DataFrame(results)
        results_df.sort_values("build", ignore_index=True, inplace=True)
        return results_df

    def run_accuracy_experiments(self, dataset_df, name, results_path):
        print("Starting NRPA experiments")
        nrpa_ranklib_ds = self.convert_to_ranklib_dataset(dataset_df)
        traning_sets_path = results_path / name / "nrpa"
        self.create_ranklib_training_sets(nrpa_ranklib_ds, traning_sets_path)
        nrpa_results = self.train_and_test("nrpa", traning_sets_path)
        nrpa_results.to_csv(traning_sets_path / "results.csv", index=False)

        failed_builds = (
            dataset_df[dataset_df[DatasetFactory.VERDICT] > 0][DatasetFactory.BUILD]
            .unique()
            .tolist()
        )
        if len(failed_builds) > 1:
            print("Starting APFD experiments")
            napfd_dataset = dataset_df[
                dataset_df[DatasetFactory.BUILD].isin(failed_builds)
            ].reset_index(drop=True)
            napfd_ranklib_ds = self.convert_to_ranklib_dataset(napfd_dataset)
            traning_sets_path = results_path / name / "napfd"
            self.create_ranklib_training_sets(napfd_ranklib_ds, traning_sets_path)
            apfd_results = self.train_and_test("napfd", traning_sets_path)
            apfd_results.to_csv(traning_sets_path / "results.csv", index=False)
        else:
            print("Not enough failed builds for APFD experiments.")

    def convert_decay_datasets(self, datasets_path, failed_build_ids):
        decay_dataset_paths = list(p for p in datasets_path.glob("*") if p.is_dir())
        for decay_dataset_path in tqdm(
            decay_dataset_paths, desc="Converting decay datasets"
        ):
            current_build_failed = int(decay_dataset_path.name) in failed_build_ids
            nrpa_test_file = decay_dataset_path / "nrpa_test.txt"
            napfd_test_file = decay_dataset_path / "napfd_test.txt"
            # Check whether converted files already exist
            if current_build_failed:
                if nrpa_test_file.exists() and napfd_test_file.exists():
                    continue
            else:
                if nrpa_test_file.exists():
                    continue

            decay_ds_df = pd.read_csv(decay_dataset_path / "dataset.csv")
            decay_ranklib_ds = self.convert_to_ranklib_dataset(decay_ds_df)
            if not nrpa_test_file.exists():
                decay_ranklib_ds.to_csv(
                    nrpa_test_file,
                    sep=" ",
                    header=False,
                    index=False,
                )
            failed_builds = (
                decay_ranklib_ds[decay_ranklib_ds["i_verdict"] > 0]["i_build"]
                .unique()
                .tolist()
            )
            if len(failed_builds) > 0 and current_build_failed:
                napfd_decay_ds = decay_ranklib_ds[
                    decay_ranklib_ds["i_build"].isin(failed_builds)
                ].reset_index(drop=True)
                if not napfd_test_file.exists():
                    napfd_decay_ds.to_csv(
                        napfd_test_file,
                        sep=" ",
                        header=False,
                        index=False,
                    )

    def test_decay_datasets(self, eval_model_paths, datasets_path, eval_metric):
        ranklib_path = Path("assets") / "RankLib.jar"
        for model_path in tqdm(eval_model_paths, desc=f"Testing {eval_metric} models"):
            model_file = model_path / "model.txt"
            test_file = datasets_path / model_path.name / f"{eval_metric}_test.txt"
            pred_file = datasets_path / model_path.name / f"{eval_metric}_pred.txt"
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
            results = {"build": [], eval_metric: []}
            for pred_build in pred_builds:
                pred_df = (
                    preds_df[preds_df["build"] == pred_build]
                    .copy()
                    .reset_index(drop=True)
                    .sort_values("score", ascending=False, ignore_index=True)
                )
                results["build"].append(pred_build)
                if eval_metric == "nrpa":
                    eval_score = self.compute_nrpa(pred_df["target"].values.tolist())
                elif eval_metric == "napfd":
                    eval_score = self.compute_napfd(pred_df)
                results[eval_metric].append(eval_score)
            pd.DataFrame(results).sort_values("build", ignore_index=True).to_csv(
                datasets_path / model_path.name / f"{eval_metric}_results.csv",
                index=False,
            )

    def run_decay_test_experiments(self, datasets_path, models_path):
        napfd_models_path = list(
            p for p in (models_path / "napfd").glob("*") if p.is_dir()
        )
        failed_build_ids = [int(model.name) for model in napfd_models_path]
        self.convert_decay_datasets(datasets_path, failed_build_ids)

        nrpa_models_path = list(
            p for p in (models_path / "nrpa").glob("*") if p.is_dir()
        )
        self.test_decay_datasets(nrpa_models_path, datasets_path, "nrpa")
        self.test_decay_datasets(napfd_models_path, datasets_path, "napfd")
