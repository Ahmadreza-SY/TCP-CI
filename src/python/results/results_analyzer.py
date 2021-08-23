import pandas as pd
from ..feature_extractor.feature import Feature
from ..entities.execution_record import ExecutionRecord
from ..entities.entity_change import EntityChange
import subprocess
import shlex
import os
from tqdm import tqdm
from pydriller import GitRepository
from .rq1_resutls_analyzer import RQ1ResultAnalyzer
from .rq2_resutls_analyzer import RQ2ResultAnalyzer
from .rq3_resutls_analyzer import RQ3ResultAnalyzer


class ResultAnalyzer:
    BUILD_THRESHOLD = 50
    DURATION_THRESHOLD = 5
    TC_THRESHOLD = 10

    def __init__(self, config):
        self.config = config
        self.subject_id_map = {}

    def analyze_results(self):
        self.generate_subject_stats_table()
        rq1_analyzer = RQ1ResultAnalyzer(self.config, self.subject_id_map)
        rq1_analyzer.analyze_results()
        rq2_analyzer = RQ2ResultAnalyzer(self.config, self.subject_id_map)
        rq2_analyzer.analyze_results()
        rq3_analyzer = RQ3ResultAnalyzer(self.config, self.subject_id_map)
        rq3_analyzer.analyze_results()

    def checkout_latest_build(self, ds_path):
        source_path = ds_path / ds_path.name.split("@")[1]
        git_repository = GitRepository(source_path)
        builds_df = pd.read_csv(ds_path / "builds.csv", parse_dates=["started_at"])
        builds_df = builds_df.sort_values(
            "started_at", ascending=False, ignore_index=True
        )
        for _, r in builds_df.iterrows():
            for last_build_commit in r["commits"].split("#"):
                try:
                    git_repository.repo.git.checkout(last_build_commit, force=True)
                    return
                except:
                    continue

    def compute_sloc(self, ds_path):
        self.checkout_latest_build(ds_path)
        source_path = ds_path / ds_path.name.split("@")[1]
        subprocess.call(
            shlex.split(
                f"cloc {source_path} --csv --out {source_path.name}.csv --timeout 120"
            ),
            stdout=subprocess.DEVNULL,
        )
        df = pd.read_csv(f"{source_path.name}.csv")
        java_sloc = df[df["language"] == "Java"].code.iloc[0]
        sum_row = df[df["language"] == "SUM"]
        sloc = sum_row.code.iloc[0] + sum_row.comment.iloc[0]
        os.remove(f"{source_path.name}.csv")
        return sloc, java_sloc

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

                if round(avg_tc) < ResultAnalyzer.TC_THRESHOLD:
                    continue

                change_history = pd.read_csv(ds_path / "entity_change_history.csv")
                builds = pd.read_csv(ds_path / "builds.csv", parse_dates=["started_at"])
                time_period = (
                    builds["started_at"].max() - builds["started_at"].min()
                ).days
                time_period = round(float(time_period) / 30.0)

                subject_stats.setdefault("Subject", []).append(ds_path.name)
                sloc, java_sloc = self.compute_sloc(ds_path)
                subject_stats.setdefault("SLOC", []).append(sloc)
                subject_stats.setdefault("Java SLOC", []).append(java_sloc)
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
        stats_path = self.config.output_path / "subject_stats.csv"
        if stats_path.exists():
            stats_df = pd.read_csv(stats_path)
        else:
            stats_df = self.extract_subjects_stats(self.config.data_path)
        if len(stats_df) == 0:
            print("No valid subject available")
            return
        stats_df.to_csv(stats_path, index=False)
        stats_df.sort_values("SLOC", ascending=False, ignore_index=True, inplace=True)
        self.subject_id_map = dict(
            zip(stats_df["Subject"].values.tolist(), (stats_df.index + 1).tolist())
        )
        stats_df["$S_{ID}$"] = stats_df["Subject"].apply(
            lambda s: f"$S_{{{self.subject_id_map[s]}}}$"
        )
        stats_df["SLOC"] = stats_df["SLOC"].apply(
            lambda n: f"{int(n/1000.0)}k"
            if n < 1000000
            else f'{float("{:.2f}".format(n/1000000.0))}M'
        )
        stats_df["Java SLOC"] = stats_df["Java SLOC"].apply(
            lambda n: f"{int(n/1000.0)}k"
            if n < 1000000
            else f'{float("{:.2f}".format(n/1000000.0))}M'
        )
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