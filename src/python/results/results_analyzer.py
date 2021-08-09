import pandas as pd
from ..feature_extractor.feature import Feature
from ..entities.execution_record import ExecutionRecord
import subprocess
import shlex
import os
from tqdm import tqdm


class ResultAnalyzer:
    BUILD_THRESHOLD = 50
    DURATION_THRESHOLD = 5

    def __init__(self, config):
        self.config = config
        self.subject_id_map = {}

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
                    else "{:.2f}".format(float(avg_duration) / 60000.0)
                )
                if avg_duration < ResultAnalyzer.DURATION_THRESHOLD:
                    continue
                subject_stats.setdefault("Subject", []).append(
                    ds_path.name.replace("@", "/")
                )
                subject_stats.setdefault("SLOC", []).append(self.compute_sloc(ds_path))
                subject_stats.setdefault("# Builds", []).append(
                    exe_df[ExecutionRecord.BUILD].nunique()
                )
                subject_stats.setdefault("# Failed Builds", []).append(failed_builds)
                subject_stats.setdefault("Avg. # TC/Build", []).append(round(avg_tc))
                subject_stats.setdefault("Avg. Test Time (min)", []).append(
                    avg_duration
                )
        results_df = pd.DataFrame(subject_stats)
        return results_df

    def generate_subject_stats_table(self):
        stats_df = self.extract_subjects_stats(self.config.data_path)
        stats_df.to_csv(self.config.output_path / "subject_stats.csv", index=False)
        stats_df.sort_values("SLOC", ascending=False, ignore_index=True, inplace=True)
        self.subject_id_map = dict(
            zip(stats_df["Subject"].values.tolist(), (stats_df.index + 1).tolist())
        )
        stats_df["$S_ID$"] = stats_df["Subject"].apply(
            lambda s: f"$S_{self.subject_id_map[s]}$"
        )
        stats_df["SLOC"] = stats_df["SLOC"].apply(lambda n: f"{int(n/1000.0)}k")
        stats_df["# Builds"] = stats_df["# Builds"].apply(lambda n: f"{n:,}")
        cols = stats_df.columns.tolist()
        cols = [cols.pop()] + cols
        stats_df = stats_df[cols]
        with (self.config.output_path / "subject_stats.tex").open("w") as f:
            f.write(stats_df.to_latex(index=False, caption="Subject statistics"))
