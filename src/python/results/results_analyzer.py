import pandas as pd
from ..feature_extractor.feature import Feature
from ..entities.execution_record import ExecutionRecord
from ..entities.entity_change import EntityChange
import subprocess
import shlex
import os
from tqdm import tqdm
import numpy as np
from scipy.stats import wilcoxon
import pingouin as pg
import matplotlib.pyplot as plt
import re
from git import Git
from pydriller import GitRepository


class ResultAnalyzer:
    BUILD_THRESHOLD = 50
    DURATION_THRESHOLD = 5
    FEATURE_GROUPS = [
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
    CORR_THRESHOLD = 0.8

    def __init__(self, config):
        self.config = config
        self.subject_id_map = {}

    def analyze_results(self):
        self.generate_subject_stats_table()
        self.generate_data_collection_time_table()
        self.generate_testing_vs_total_time_table()
        self.generate_time_tests_table()
        self.generate_total_vs_impacted_time_table()
        self.generate_time_subject_corr()

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
        stats_df["SLOC"] = stats_df["SLOC"].apply(lambda n: f"{int(n/1000.0)}k")
        stats_df["Java SLOC"] = stats_df["Java SLOC"].apply(
            lambda n: f"{int(n/1000.0)}k"
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

    def compute_data_collection_time(self):
        summary = {}
        summary["S_ID"] = []
        for fg in ResultAnalyzer.FEATURE_GROUPS:
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
            for fg in ResultAnalyzer.FEATURE_GROUPS:
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
        for fg in ResultAnalyzer.FEATURE_GROUPS:
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

    def run_time_statistical_tests(self):
        subjects_time_df_list = []
        for subject, sid in self.subject_id_map.items():
            time_df = pd.read_csv(
                self.config.data_path / subject / "feature_group_time.csv"
            )
            time_df["Build"] = f"{sid}-" + time_df["Build"].astype(str)
            subjects_time_df_list.append(time_df)

        subjects_time_df = pd.concat(subjects_time_df_list, ignore_index=True)

        fg_time_df = subjects_time_df.pivot_table(
            index="Build", columns="FeatureGroup", values="TotalTime"
        )
        wilcoxon_res = {"A": [], "B": [], "p-value": [], "CL": []}
        for i, f1 in enumerate(ResultAnalyzer.FEATURE_GROUPS):
            for f2 in ResultAnalyzer.FEATURE_GROUPS[i + 1 :]:
                x, y = fg_time_df[f1].values, fg_time_df[f2].values
                z, p = wilcoxon(x, y)
                wilcoxon_res["A"].append(f1.replace("_", "\\_"))
                wilcoxon_res["B"].append(f2.replace("_", "\\_"))
                wilcoxon_res["p-value"].append(p)
                cl = pg.compute_effsize(x, y, paired=True, eftype="CLES")
                wilcoxon_res["CL"].append(float("{:.2f}".format(cl)))
        results = pd.DataFrame(wilcoxon_res).sort_values(
            ["A", "B", "p-value"], ignore_index=True
        )
        fg_cnt = dict(results.groupby("A").count()["B"].items())
        results["count"] = results["A"].apply(lambda a: fg_cnt[a])
        results.sort_values(
            ["count", "p-value", "CL"],
            ascending=[False, True, False],
            ignore_index=True,
            inplace=True,
        )
        results.drop("count", axis=1, inplace=True)
        return results

    def generate_time_tests_table(self):
        results = self.run_time_statistical_tests()
        results.to_csv(self.config.output_path / "rq1_test_results.csv", index=False)
        results["p-value"] = results["p-value"].apply(
            lambda p: "$<0.01$" if 0.0 < p < 0.01 else "{:.2f}".format(p)
        )
        index_tuples = results.apply(lambda r: (r["A"], r["B"]), axis=1).values.tolist()
        results.set_index(pd.MultiIndex.from_tuples(index_tuples), inplace=True)
        results.drop(["A", "B"], axis=1, inplace=True)
        with (self.config.output_path / "rq1_test_results.tex").open("w") as f:
            f.write(results.to_latex(index=True, escape=False, multirow=True))

    def compute_total_vs_impacted_time(self):
        results = {"s": [], "adct": [], "aict": [], "ic": []}
        for subject, sid in tqdm(
            self.subject_id_map.items(), desc="Computing total vs impacted times"
        ):
            results["s"].append(sid)
            output_path = self.config.data_path / subject
            imp_time_df = pd.read_csv(output_path / "impacted_time.csv")
            total_time_avg = (
                imp_time_df[imp_time_df["ProcessName"] == "Total"][
                    Feature.DURATION
                ].mean()
                / 60.0
            )
            impacted_time_avg = (
                imp_time_df[imp_time_df["ProcessName"] == "Impacted"][
                    Feature.DURATION
                ].mean()
                / 60.0
            )
            results["adct"].append(total_time_avg)
            results["aict"].append(impacted_time_avg)

            ic = round((impacted_time_avg * 100.0) / total_time_avg)
            results["ic"].append(ic)

        results_df = pd.DataFrame(results)
        return results_df

    def generate_total_vs_impacted_time_table(self):
        time_df = self.compute_total_vs_impacted_time()
        time_df.to_csv(
            self.config.output_path / "rq1_total_vs_impacted_time.csv", index=False
        )
        time_df["s"] = time_df["s"].apply(lambda id: f"$S_{{{id}}}$")
        time_df["aict"] = time_df["aict"].apply(lambda n: "{:.1f}".format(n))
        time_df["adct"] = time_df["adct"].apply(lambda n: "{:.1f}".format(n))
        time_df.columns = [
            "$S_{ID}$",
            "Avg. Total Collection Time",
            "Avg. Impacted Collection Time",
            "Impacted/Total (\\%)",
        ]
        with (self.config.output_path / "rq1_total_vs_impacted_time.tex").open(
            "w"
        ) as f:
            f.write(time_df.to_latex(index=False, escape=False))

    def compute_time_subject_corr(self):
        stats_cols = [
            "SLOC",
            "Java SLOC",
            "\\# Commits",
            "\\# Builds",
            "\\# Failed Builds",
            "Avg. \\# TC/Build",
            "Avg. Test Time (min)",
        ]
        fg_time_df = pd.read_csv(self.config.output_path / "rq1_avg_time.csv")
        fg_time_df = fg_time_df[fg_time_df["S_ID"] != "Avg"]
        fg_time_df["S_ID"] = fg_time_df["S_ID"].astype(int)
        total_time_df = pd.read_csv(
            self.config.output_path / "rq1_total_vs_impacted_time.csv"
        )
        total_time_df["S_ID"] = total_time_df["s"]
        total_time_df["Total"] = total_time_df["adct"]
        fg_time_df = fg_time_df.merge(total_time_df[["S_ID", "Total"]], on="S_ID")
        for fg in ResultAnalyzer.FEATURE_GROUPS:
            col = f"{fg}-T"
            fg_time_df[fg] = fg_time_df[col]
        fg_time_df = fg_time_df[["S_ID"] + ResultAnalyzer.FEATURE_GROUPS + ["Total"]]
        stats_df = pd.read_csv(self.config.output_path / "subject_stats.csv")
        stats_df["S_ID"] = stats_df["Subject"].apply(lambda s: self.subject_id_map[s])
        data = stats_df.merge(fg_time_df, on="S_ID")
        data.drop(["Subject", "S_ID"], axis=1, inplace=True)
        corr_df = data.corr(method="pearson")
        corr_results = {"s": []}
        for col in stats_cols:
            corr_results["s"].append(col)
            selected_corr = (
                corr_df[col][ResultAnalyzer.FEATURE_GROUPS + ["Total"]]
                .abs()
                .sort_values(ascending=False)
            )
            for fg, corr_val in selected_corr.iteritems():
                corr_results.setdefault(fg, []).append(corr_val)

        return pd.DataFrame(corr_results), data

    def generate_time_subject_corr(self):
        corr_results, data = self.compute_time_subject_corr()
        corr_results.to_csv(
            self.config.output_path / "rq1_size_time_corr.csv", index=False
        )
        corr_results["Charac."] = corr_results["s"]
        corr_results = corr_results[
            ["Charac."] + ResultAnalyzer.FEATURE_GROUPS + ["Total"]
        ]

        for col in ResultAnalyzer.FEATURE_GROUPS + ["Total"]:
            corr_results[col] = corr_results[col].apply(
                lambda n: "\\textbf{{{:.2f}}}".format(n)
                if n > ResultAnalyzer.CORR_THRESHOLD
                else "{:.2f}".format(n)
            )
        corr_results.columns = list(
            map(lambda c: c.replace("_", "\\_"), list(corr_results.columns))
        )

        with (self.config.output_path / "rq1_size_time_corr.tex").open("w") as f:
            f.write(corr_results.to_latex(index=False, escape=False))

        def plot_corr(f1, f2, ax):
            x = data[f1].values
            y = data[f2].values
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, y, "bo")
            ax.plot([min(x), max(x)], [p(min(x)), p(max(x))], "r--")
            ax.set(xlabel=f1, ylabel=f2 + " (sec.)")

        fig, axs = plt.subplots(2, 3, figsize=(24, 9))
        plot_corr("SLOC", "Total", axs[0, 0])
        plot_corr("SLOC", "COV", axs[0, 1])
        plot_corr("SLOC", "COD_COV_COM", axs[0, 2])
        plot_corr("Avg. \\# TC/Build", "Total", axs[1, 0])
        plot_corr("Avg. \\# TC/Build", "COV", axs[1, 1])
        plot_corr("Avg. \\# TC/Build", "REC", axs[1, 2])
        plt.savefig(
            self.config.output_path / "rq1_corr_analysis.png", bbox_inches="tight"
        )
