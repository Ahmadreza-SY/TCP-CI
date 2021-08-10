import pandas as pd
from ..feature_extractor.feature import Feature
from ..entities.execution_record import ExecutionRecord
from ..entities.entity_change import EntityChange
import subprocess
import shlex
import os
from tqdm import tqdm
import numpy as np


class ResultAnalyzer:
    BUILD_THRESHOLD = 50
    DURATION_THRESHOLD = 5

    def __init__(self, config):
        self.config = config
        self.subject_id_map = {}

    def analyze_results(self):
        self.generate_subject_stats_table()
        self.generate_data_collection_time_table()
        self.generate_testing_vs_total_time_table()

    def compute_sloc(self, ds_path):
        source_path = ds_path / ds_path.name.split("@")[1]
        subprocess.call(
            shlex.split(
                f"cloc {source_path} --csv --out {source_path.name}.csv --timeout 120"
            ),
            stdout=subprocess.DEVNULL,
        )
        df = pd.read_csv(f"{source_path.name}.csv")
        loc = df[df["language"] == "Java"].code.iloc[0]
        os.remove(f"{source_path.name}.csv")
        return loc

    def extract_subjects_stats(self, data_path):
        ds_paths = [p for p in data_path.glob("*") if p.is_dir()]
        subject_stats = {}
        for ds_path in tqdm(ds_paths, desc="Computing subject stats"):
            features_path = ds_path / "dataset.csv"
            if features_path.exists():
                ds_df = pd.read_csv(features_path)
                failed_builds = ds_df[Feature.BUILD].nunique()
                if failed_builds < ResultAnalyzer.BUILD_THRESHOLD:
                    continue

                exe_df = pd.read_csv(ds_path / "exe.csv")
                avg_tc = (
                    exe_df.groupby(ExecutionRecord.BUILD)
                    .count()[ExecutionRecord.TEST]
                    .mean()
                )
                avg_duration = (
                    exe_df.groupby(ExecutionRecord.BUILD)
                    .sum()[ExecutionRecord.DURATION]
                    .mean()
                )
                avg_duration = (
                    int(avg_duration / 60000.0)
                    if avg_duration > 60000
                    else float("{:.2f}".format(float(avg_duration) / 60000.0))
                )
                if avg_duration < ResultAnalyzer.DURATION_THRESHOLD:
                    continue

                change_history = pd.read_csv(ds_path / "entity_change_history.csv")
                builds = pd.read_csv(ds_path / "builds.csv", parse_dates=["started_at"])
                time_period = (
                    builds["started_at"].max() - builds["started_at"].min()
                ).days
                time_period = round(float(time_period) / 30.0)

                subject_stats.setdefault("Subject", []).append(ds_path.name)
                subject_stats.setdefault("SLOC", []).append(self.compute_sloc(ds_path))
                subject_stats.setdefault("\\# Commits", []).append(
                    change_history[EntityChange.COMMIT].nunique()
                )
                subject_stats.setdefault("Time period (months)", []).append(time_period)
                subject_stats.setdefault("\\# Builds", []).append(
                    exe_df[ExecutionRecord.BUILD].nunique()
                )
                subject_stats.setdefault("\\# Failed Builds", []).append(failed_builds)
                subject_stats.setdefault("Avg. \\# TC/Build", []).append(round(avg_tc))
                subject_stats.setdefault("Avg. Test Time (min)", []).append(
                    avg_duration
                )
        results_df = pd.DataFrame(subject_stats)
        return results_df

    def generate_subject_stats_table(self):
        stats_df = self.extract_subjects_stats(self.config.data_path)
        if len(stats_df) == 0:
            print("No valid subject available")
            return
        stats_df.to_csv(self.config.output_path / "subject_stats.csv", index=False)
        stats_df.sort_values("SLOC", ascending=False, ignore_index=True, inplace=True)
        self.subject_id_map = dict(
            zip(stats_df["Subject"].values.tolist(), (stats_df.index + 1).tolist())
        )
        stats_df["$S_{ID}$"] = stats_df["Subject"].apply(
            lambda s: f"$S_{{{self.subject_id_map[s]}}}$"
        )
        stats_df["SLOC"] = stats_df["SLOC"].apply(lambda n: f"{int(n/1000.0)}k")
        stats_df["\\# Builds"] = stats_df["\\# Builds"].apply(lambda n: f"{n:,}")
        stats_df["\\# Commits"] = stats_df["\\# Commits"].apply(
            lambda n: "{:.1f}".format(n / 1000.0) + "k"
        )
        stats_df["Subject"] = stats_df["Subject"].apply(lambda s: s.replace("@", "/"))
        cols = stats_df.columns.tolist()
        cols = [cols.pop()] + cols
        stats_df = stats_df[cols]
        with (self.config.output_path / "subject_stats.tex").open("w") as f:
            f.write(
                stats_df.to_latex(
                    index=False, caption="Subject statistics", escape=False
                )
            )

    def compute_data_collection_time(self):
        feature_groups = [
            "COD_COV_COM",
            "COD_COV_PRO",
            "DET_COV",
            "COD_COV_CHN",
            "COV",
            "TES_COM",
            "TES_PRO",
            "REC",
            "TES_CHN",
        ]
        summary = {}
        summary["S_ID"] = []
        for fg in feature_groups:
            summary[f"{fg}-P"] = []
            summary[f"{fg}-M"] = []
            summary[f"{fg}-T"] = []
        all_subjects_time_df = []

        for subject, sid in tqdm(
            self.subject_id_map.items(), desc="Computing avg data collection times"
        ):
            time_df = pd.read_csv(
                self.config.data_path / subject / "feature_group_time.csv"
            )
            all_subjects_time_df.append(time_df)
            summary["S_ID"].append(sid)
            for fg in feature_groups:
                p_time = time_df[time_df["FeatureGroup"] == fg][
                    "PreprocessingTime"
                ].values.tolist()
                m_time = time_df[time_df["FeatureGroup"] == fg][
                    "MeasurementTime"
                ].values.tolist()
                t_time = time_df[time_df["FeatureGroup"] == fg][
                    "TotalTime"
                ].values.tolist()
                summary[f"{fg}-P"].append(np.mean(p_time))
                summary[f"{fg}-M"].append(np.mean(m_time))
                summary[f"{fg}-T"].append(np.mean(t_time))

        g_time_df = pd.concat(all_subjects_time_df)
        summary["S_ID"].append("Avg")
        for fg in feature_groups:
            p_time = g_time_df[g_time_df["FeatureGroup"] == fg][
                "PreprocessingTime"
            ].values.tolist()
            m_time = g_time_df[g_time_df["FeatureGroup"] == fg][
                "MeasurementTime"
            ].values.tolist()
            t_time = g_time_df[g_time_df["FeatureGroup"] == fg][
                "TotalTime"
            ].values.tolist()
            summary[f"{fg}-P"].append(np.mean(p_time))
            summary[f"{fg}-M"].append(np.mean(m_time))
            summary[f"{fg}-T"].append(np.mean(t_time))

        return pd.DataFrame(summary)

    def generate_data_collection_time_table(self):
        time_results = self.compute_data_collection_time()
        time_results.to_csv(self.config.output_path / "rq1_avg_time.csv", index=False)
        for col in time_results.columns.tolist():
            if col == "S_ID":
                continue
            time_results[col] = time_results[col].apply(lambda n: "{:.1f}".format(n))
        time_results["$S_{ID}$"] = time_results["S_ID"].apply(
            lambda id: f"$S_{{{id}}}$" if id != "Avg" else id
        )
        time_results.drop("S_ID", axis=1, inplace=True)
        cols = time_results.columns.tolist()
        cols = [cols.pop()] + cols
        time_results = time_results[cols]
        with (self.config.output_path / "rq1_avg_time.tex").open("w") as f:
            f.write(time_results.to_latex(index=False, escape=False))

    def compute_testing_vs_total_time(self):
        results = {"s": [], "att": [], "adct": [], "ct": []}
        for subject, sid in tqdm(
            self.subject_id_map.items(), desc="Computing testing vs total times"
        ):
            results["s"].append(sid)
            output_path = self.config.data_path / subject
            exe_df = pd.read_csv(output_path / "exe.csv")
            avg_testing_time = (
                exe_df.groupby(ExecutionRecord.BUILD)
                .sum()[ExecutionRecord.DURATION]
                .mean()
                / 60000.0
            )
            results["att"].append(avg_testing_time)

            imp_time_df = pd.read_csv(output_path / "impacted_time.csv")
            total_time_avg = (
                imp_time_df[imp_time_df["ProcessName"] == "Total"][
                    Feature.DURATION
                ].mean()
                / 60.0
            )
            results["adct"].append(total_time_avg)

            ct = round((total_time_avg * 100.0) / avg_testing_time)
            results["ct"].append(ct)

        results_df = pd.DataFrame(results)
        return results_df

    def generate_testing_vs_total_time_table(self):
        time_df = self.compute_testing_vs_total_time()
        time_df.to_csv(
            self.config.output_path / "rq1_testing_vs_total_time.csv", index=False
        )
        time_df["s"] = time_df["s"].apply(lambda id: f"$S_{{{id}}}$")
        time_df["att"] = time_df["att"].apply(lambda n: "{:.1f}".format(n))
        time_df["adct"] = time_df["adct"].apply(lambda n: "{:.1f}".format(n))
        time_df.columns = [
            "$S_{ID}$",
            "Avg. Testing Time",
            "Avg. Data Collection Time",
            "Collection/Testing (\\%)",
        ]
        with (self.config.output_path / "rq1_testing_vs_total_time.tex").open("w") as f:
            f.write(time_df.to_latex(index=False, escape=False))
