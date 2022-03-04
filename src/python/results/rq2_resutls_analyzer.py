import pandas as pd
from scipy.stats import wilcoxon
import pingouin as pg
import matplotlib.pyplot as plt
from .rq1_resutls_analyzer import RQ1ResultAnalyzer
from ..feature_extractor.feature import Feature
import seaborn as sns
from tqdm import tqdm
import numpy as np
from scipy.stats import friedmanchisquare
import matplotlib


class RQ2ResultAnalyzer:
    SELECTED_HEURISTIC = "56-dsc"

    def __init__(self, config, subject_id_map):
        self.config = config
        self.subject_id_map = subject_id_map
        self.outliers = True

    def analyze_results(self):
        self.generate_accuracy_avg_tables()
        self.generate_feature_group_acc_test_tables()
        self.generate_feature_usage_freq_table()
        self.generate_tc_age_histogram()
        self.generate_heuristic_comparison_table()
        self.generate_apfdc_corr_tables()
        # self.generate_outliers_stats()
        self.generate_rankers_avg_table()

    def get_output_path(self):
        output_path = self.config.output_path / "RQ2"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def compute_accuracy_avg(self, metric):
        results = {}
        for subject, sid in self.subject_id_map.items():
            results.setdefault("S_ID", []).append(sid)
            res_path = self.config.data_path / subject / "tsp_accuracy_results"
            exp_paths = [p for p in res_path.glob("*") if p.is_dir()]
            for exp_path in exp_paths:
                exp = exp_path.name
                eval_df = pd.read_csv(exp_path / "results.csv")
                results.setdefault(exp, []).append(eval_df[metric].mean())
                results.setdefault(f"{exp}-std", []).append(eval_df[metric].std())
        return pd.DataFrame(results)

    def generate_accuracy_avg_table(self, results):
        col_suffix = "-outliers" if self.outliers else ""

        results = results.sort_values(
            "full" + col_suffix, ascending=False, ignore_index=True
        )
        results["$S_{ID}$"] = results["S_ID"].apply(lambda id: f"$S_{{{id}}}$")
        results["Full-inc"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                r["full" + col_suffix], r[f'{"full" + col_suffix}-std']
            ),
            axis=1,
        )
        results["Impacted-ex"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                r["wo-impacted" + col_suffix], r[f'{"wo-impacted" + col_suffix}-std']
            ),
            axis=1,
        )
        results["\\textit{TES\\_M}"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                r["W-Code" + col_suffix], r[f"{'W-Code' + col_suffix}-std"]
            ),
            axis=1,
        )
        results["\\textit{REC\\_M}"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                r["W-Execution" + col_suffix], r[f"{'W-Execution' + col_suffix}-std"]
            ),
            axis=1,
        )
        results["\\textit{COV\\_M}"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                r["W-Coverage" + col_suffix], r[f"{'W-Coverage' + col_suffix}-std"]
            ),
            axis=1,
        )
        return results[
            [
                "$S_{ID}$",
                "Full-inc",
                "Impacted-ex",
                "\\textit{TES\\_M}",
                "\\textit{REC\\_M}",
                "\\textit{COV\\_M}",
            ]
        ]

    def generate_accuracy_avg_tables(self):
        apfd = self.compute_accuracy_avg("apfd")
        apfdc = self.compute_accuracy_avg("apfdc")
        apfd.to_csv(self.get_output_path() / "rq2_apfd_avg.csv", index=False)
        apfdc.to_csv(self.get_output_path() / "rq2_apfdc_avg.csv", index=False)
        with (self.get_output_path() / "rq2_apfd_avg.tex").open("w") as f:
            f.write(
                self.generate_accuracy_avg_table(apfd).to_latex(
                    index=False, escape=False
                )
            )
        with (self.get_output_path() / "rq2_apfdc_avg.tex").open("w") as f:
            f.write(
                self.generate_accuracy_avg_table(apfdc).to_latex(
                    index=False, escape=False
                )
            )

    def run_feature_group_acc_tests(self, metric, experiments):
        col_suffix = "-outliers" if self.outliers else ""
        samples = {}
        for subject, _ in self.subject_id_map.items():
            results_df = pd.read_csv(
                self.config.data_path
                / subject
                / "tsp_accuracy_results"
                / ("full" + col_suffix)
                / "results.csv"
            ).sort_values("build", ignore_index=True)
            samples.setdefault("build", []).extend(results_df["build"].values.tolist())
            samples.setdefault("full" + col_suffix, []).extend(
                results_df[metric].values.tolist()
            )
            for exp in experiments:
                results_df = pd.read_csv(
                    self.config.data_path
                    / subject
                    / "tsp_accuracy_results"
                    / exp
                    / "results.csv"
                ).sort_values("build", ignore_index=True)
                samples.setdefault(exp, []).extend(results_df[metric].values.tolist())
        samples_df = pd.DataFrame(samples)

        wilcoxon_res = {"A": [], "B": [], "p-value": [], "CL": []}
        for exp in experiments:
            x, y = samples_df["full" + col_suffix].values, samples_df[exp].values
            z, p = wilcoxon(x, y)
            wilcoxon_res["A"].append("full" + col_suffix)
            wilcoxon_res["B"].append(exp)
            wilcoxon_res["p-value"].append(p)
            cl = pg.compute_effsize(x, y, paired=True, eftype="CLES")
            wilcoxon_res["CL"].append(cl)
        return pd.DataFrame(wilcoxon_res).sort_values("p-value", ignore_index=True)

    def generate_feature_group_acc_test_table(self, experiments, name):
        col_suffix = "-outliers" if self.outliers else ""
        apfd = self.run_feature_group_acc_tests("apfd", experiments)
        apfdc = self.run_feature_group_acc_tests("apfdc", experiments)
        apfd.to_csv(self.get_output_path() / f"rq2_apfd_{name}_tests.csv", index=False)
        apfdc.to_csv(
            self.get_output_path() / f"rq2_apfdc_{name}_tests.csv", index=False
        )

        def format_columns(df):
            name_dict = {
                ("W-Code" + col_suffix): "TES_M",
                ("W-Execution" + col_suffix): "REC_M",
                ("W-Coverage" + col_suffix): "COV_M",
            }
            df["B"] = df["B"].apply(lambda v: name_dict[v] if v in name_dict else v)
            df["B"] = df["B"].apply(lambda v: v.replace("_", "\\_").replace("wo-", ""))
            df["A"] = df["A"].apply(lambda v: v.replace(col_suffix, "").title())
            df["p-value"] = df["p-value"].apply(
                lambda p: "$<0.01$"
                if 0.0 < p < 0.01
                else (
                    "{:.3f}".format(p)
                    if float("{:.2f}".format(p)) == 0.05
                    else "{:.2f}".format(p)
                )
            )
            df["CL"] = df["CL"].apply(lambda cl: "{:.2f}".format(cl))
            return df

        apfd = format_columns(apfd)
        apfdc = format_columns(apfdc)
        # tests_df = apfd.merge(apfdc, on=["A", "B"], suffixes=["-apfd", "-apfdc"])
        with (self.get_output_path() / f"rq2_{name}_tests.tex").open("w") as f:
            f.write(apfdc.to_latex(index=False, escape=False))

    def generate_feature_group_acc_test_tables(self):
        col_suffix = "-outliers" if self.outliers else ""
        without_experiments = [
            f"wo-{fg}" + col_suffix for fg in RQ1ResultAnalyzer.FEATURE_GROUPS
        ]
        self.generate_feature_group_acc_test_table(without_experiments, "without")

        with_experiments = [
            "W-Code" + col_suffix,
            "W-Execution" + col_suffix,
            "W-Coverage" + col_suffix,
        ]
        self.generate_feature_group_acc_test_table(with_experiments, "with")

        impacted_experiment = ["wo-impacted" + col_suffix]
        apfd = self.run_feature_group_acc_tests("apfd", impacted_experiment)
        apfdc = self.run_feature_group_acc_tests("apfdc", impacted_experiment)
        apfd.to_csv(self.get_output_path() / f"rq2_apfd_imp_test.csv", index=False)
        apfdc.to_csv(self.get_output_path() / f"rq2_apfdc_imp_test.csv", index=False)

    def compute_avg_feature_usage_freq(self):
        col_suffix = "-outliers" if self.outliers else ""
        freq_samples = {}
        for subject, _ in tqdm(
            self.subject_id_map.items(), desc="Computing usage frequencies"
        ):
            fid_map_df = pd.read_csv(
                self.config.data_path / subject / "feature_id_map.csv"
            )
            fid_map = dict(
                zip(
                    fid_map_df["value"].values.tolist(),
                    fid_map_df["key"].values.tolist(),
                )
            )
            results_path = [
                p
                for p in (
                    self.config.data_path
                    / subject
                    / "tsp_accuracy_results"
                    / ("full" + col_suffix)
                ).glob("*")
                if p.is_dir()
            ]
            for path in results_path:
                freq_df = pd.read_csv(path / "feature_stats.csv")
                for _, r in freq_df.iterrows():
                    fname = fid_map[r["feature_id"]]
                    freq_samples.setdefault(fname, []).append(r["frequency"])
        return list(pd.DataFrame(freq_samples).mean().items())

    def generate_feature_usage_freq_table(self):
        avg_usage_freq = self.compute_avg_feature_usage_freq()
        avg_usage_freq.sort(key=lambda t: t[1], reverse=True)
        avg_usage_freq_df = pd.DataFrame(
            {
                "feature_name": [t[0] for t in avg_usage_freq],
                "usage_frequency_avg": [t[1] for t in avg_usage_freq],
            }
        )
        avg_usage_freq_df.to_csv(
            self.get_output_path() / f"rq2_avg_usage_freq.csv", index=False
        )
        avg_usage_freq = avg_usage_freq[:15]
        result = {
            "Feature Group": [
                Feature.get_feature_group(name).replace("_", "\\_")
                for name, _ in avg_usage_freq
            ],
            "Feature": [name.split("_")[-1] for name, _ in avg_usage_freq],
            "Freq. Avg.": [int(avg) for _, avg in avg_usage_freq],
        }
        result_df = pd.DataFrame(result)
        with (self.get_output_path() / f"rq2_avg_usage_freq.tex").open("w") as f:
            f.write(result_df.to_latex(index=False, escape=False))

    def generate_tc_age_histogram(self):
        x = []
        for subject, _ in tqdm(
            self.subject_id_map.items(), desc="Computing age historgram"
        ):
            ds_df = pd.read_csv(self.config.data_path / subject / "dataset.csv")
            max_age = {}
            for test in ds_df["Test"].unique():
                max_age[test] = ds_df[ds_df["Test"] == test]["REC_Age"].max()
            ds_df["Normalized_Age"] = [
                int((r["REC_Age"] * 100) / max_age[r["Test"]])
                if max_age[r["Test"]] > 0
                else 0
                for _, r in ds_df.iterrows()
            ]
            failed_norm_ages = ds_df[ds_df["Verdict"] > 0][
                "Normalized_Age"
            ].values.tolist()
            x.extend(failed_norm_ages)

        font = {"size": 16}
        matplotlib.rc("font", **font)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
        ax.set(xlabel="Normalized Age", ylabel="Failure Count")
        sns.histplot(x, ax=ax, kde=False, bins=10, color="blue")
        plt.savefig(self.get_output_path() / "rq2_tc_age_hist.png", bbox_inches="tight")

    def run_heuristic_tests(
        self, metric, experiment, heuristic=None, run_all=False, ml_experiment=None
    ):
        results = {}
        all_h_acc = []
        all_full_acc = []
        for subject, sid in self.subject_id_map.items():
            results.setdefault("S_ID", []).append(sid)
            fid_map_df = pd.read_csv(
                self.config.data_path / subject / "feature_id_map.csv"
            )
            fid_map = dict(
                zip(
                    fid_map_df["value"].values.tolist(),
                    fid_map_df["key"].values.tolist(),
                )
            )
            h_df = pd.read_csv(
                self.config.data_path
                / subject
                / "tsp_accuracy_results"
                / experiment
                / f"heuristic_{metric}_results.csv"
            ).sort_values("build", ignore_index=True)
            ml_experiment = experiment if ml_experiment is None else ml_experiment
            full_df = pd.read_csv(
                self.config.data_path
                / subject
                / "tsp_accuracy_results"
                / ml_experiment
                / "results.csv"
            ).sort_values("build", ignore_index=True)
            test_builds = full_df["build"].values.tolist()
            test_h_df = (
                h_df[h_df["build"].isin(test_builds)]
                .copy()
                .reset_index(drop=True)
                .sort_values("build", ignore_index=True)
            )
            if heuristic is not None:
                sel_fid = heuristic
            else:
                train_h_df = (
                    h_df[~h_df["build"].isin(test_builds)].copy().reset_index(drop=True)
                )
                sel_fid = train_h_df.drop("build", axis=1).mean().idxmax()
            sel_fname = fid_map[int(sel_fid.split("-")[0])]
            results.setdefault("feature", []).append(sel_fname)
            results.setdefault("full_avg", []).append(full_df[metric].mean())
            results.setdefault("full_std", []).append(full_df[metric].std())
            results.setdefault("h_avg", []).append(test_h_df[sel_fid].mean())
            results.setdefault("h_std", []).append(test_h_df[sel_fid].std())
            x, y = full_df[metric].values, test_h_df[sel_fid].values
            all_full_acc.extend(x)
            all_h_acc.extend(y)
            z, p = wilcoxon(x, y)
            cl = pg.compute_effsize(x, y, paired=True, eftype="CLES")
            results.setdefault("p-value", []).append(p)
            results.setdefault("CL", []).append(cl)

        results = pd.DataFrame(results).sort_values(
            "CL", ascending=False, ignore_index=True
        )
        if run_all:
            z, p = wilcoxon(all_full_acc, all_h_acc)
            cl = pg.compute_effsize(x, y, paired=True, eftype="CLES")
            all_results = pd.DataFrame(
                {
                    "full_avg": [np.mean(all_full_acc)],
                    "full_std": [np.std(all_full_acc)],
                    "h_avg": [np.mean(all_h_acc)],
                    "h_std": [np.std(all_h_acc)],
                    "p-value": [p],
                    "CL": [cl],
                }
            )
            return results, all_results

        return results

    def generate_heuristic_comparison_table(self):
        col_suffix = "-outliers" if self.outliers else ""
        apfd = self.run_heuristic_tests("apfd", "full" + col_suffix)
        apfdc = self.run_heuristic_tests("apfdc", "full" + col_suffix)
        apfd.to_csv(
            self.get_output_path() / f"rq2_apfd_heuristic_comp.csv", index=False
        )
        apfdc.to_csv(
            self.get_output_path() / f"rq2_apfdc_heuristic_comp.csv", index=False
        )

        apfd_custom = self.run_heuristic_tests(
            "apfd", "full" + col_suffix, heuristic=RQ2ResultAnalyzer.SELECTED_HEURISTIC
        )
        apfdc_custom, apfdc_custom_all = self.run_heuristic_tests(
            "apfdc",
            "full" + col_suffix,
            heuristic=RQ2ResultAnalyzer.SELECTED_HEURISTIC,
            run_all=True,
        )
        apfd_custom.to_csv(
            self.get_output_path() / f"rq2_apfd_56dsc_heuristic.csv", index=False
        )
        apfdc_custom.to_csv(
            self.get_output_path() / f"rq2_apfdc_56dsc_heuristic.csv", index=False
        )
        apfdc_custom_all.to_csv(
            self.get_output_path() / f"rq2_apfdc_all_56dsc_heuristic.csv", index=False
        )

        apfdc_rec, apfdc_rec_all = self.run_heuristic_tests(
            "apfdc",
            "full" + col_suffix,
            heuristic=RQ2ResultAnalyzer.SELECTED_HEURISTIC,
            run_all=True,
            ml_experiment="W-Execution" + col_suffix,
        )

        apfdc_rec_all.to_csv(
            self.get_output_path() / f"rq2_apfdc_all_56dsc_rec_heuristic.csv",
            index=False,
        )

        apfdc_tes, apfdc_tes_all = self.run_heuristic_tests(
            "apfdc",
            "full" + col_suffix,
            heuristic=RQ2ResultAnalyzer.SELECTED_HEURISTIC,
            run_all=True,
            ml_experiment="W-Code" + col_suffix,
        )

        apfdc_tes_all.to_csv(
            self.get_output_path() / f"rq2_apfdc_all_56dsc_tes_heuristic.csv",
            index=False,
        )

        def format_columns(df):
            df["$S_{ID}$"] = df["S_ID"].apply(lambda id: f"$S_{{{id}}}$")
            df["Best H Feature"] = df["feature"].apply(lambda f: f.replace("_", "\\_"))
            df["Best H"] = df.apply(
                lambda r: "${:.2f} {{\pm}} {:.2f}$".format(r["h_avg"], r["h_std"]),
                axis=1,
            )
            df["\\textit{Full\\_M}"] = df.apply(
                lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                    r["full_avg"], r["full_std"]
                ),
                axis=1,
            )
            df["p-value"] = df["p-value"].apply(
                lambda p: "$<0.01$"
                if 0.0 < p < 0.01
                else (
                    "{:.3f}".format(p)
                    if float("{:.2f}".format(p)) == 0.05
                    else "{:.2f}".format(p)
                )
            )
            df["CL"] = df["CL"].apply(lambda cl: "{:.2f}".format(cl))
            return (
                df[
                    [
                        "$S_{ID}$",
                        "\\textit{Full\\_M}",
                        "Best H Feature",
                        "Best H",
                        "p-value",
                        "CL",
                    ]
                ]
                .copy()
                .reset_index(drop=True)
            )

        with (self.get_output_path() / f"rq2_apfdc_heuristic_comp.tex").open("w") as f:
            f.write(format_columns(apfdc).to_latex(index=False, escape=False))

        def customize_cols(df):
            df["\\textit{H\\_M}"] = df["Best H"]
            df.drop(["Best H Feature", "Best H"], axis=1, inplace=True)
            return df[
                [
                    "$S_{ID}$",
                    "\\textit{Full\\_M}",
                    "\\textit{H\\_M}",
                    "p-value",
                    "CL",
                ]
            ]

        with (self.get_output_path() / f"rq2_apfdc_56dsc_heuristic.tex").open("w") as f:
            apfdc_custom_res = customize_cols(format_columns(apfdc_custom))
            apfdc_custom_res.sort_values(
                "CL", ascending=False, ignore_index=True, inplace=True
            )
            f.write(apfdc_custom_res.to_latex(index=False, escape=False))

    def compute_subject_corr(self, target_df, target_cols):
        stats_cols = [
            "SLOC",
            "Java SLOC",
            "\\# Commits",
            "\\# Builds",
            "\\# Failed Builds",
            "Failure Rate (%)",
            "Avg. \\# TC/Build",
            "Avg. Test Time (min)",
        ]
        stats_df = pd.read_csv(self.config.output_path / "subject_stats.csv")
        stats_df["S_ID"] = stats_df["Subject"].apply(lambda s: self.subject_id_map[s])
        data = stats_df.merge(target_df, on="S_ID")
        data = data[target_cols + stats_cols]
        corr_df = data.corr(method="spearman")
        corr_results = {"s": []}
        for col in stats_cols:
            corr_results["s"].append(col)
            selected_corr = corr_df[col][target_cols].sort_values(ascending=False)
            for tr, corr_val in selected_corr.iteritems():
                corr_results.setdefault(tr, []).append(corr_val)

        return pd.DataFrame(corr_results), data

    def generate_apfdc_corr_tables(self):
        col_suffix = "-outliers" if self.outliers else ""
        avg_apfdc_df = pd.read_csv(self.get_output_path() / "rq2_apfdc_avg.csv")
        avg_apfdc_df["S_ID"] = avg_apfdc_df["S_ID"].astype(int)
        corr_results, data = self.compute_subject_corr(
            avg_apfdc_df, ["full" + col_suffix]
        )
        corr_results.to_csv(
            self.get_output_path() / "rq2_size_apfdc_corr.csv", index=False
        )

        h_df = pd.read_csv(self.get_output_path() / "rq2_apfdc_56dsc_heuristic.csv")
        h_df["S_ID"] = h_df["S_ID"].astype(int)
        h_df["ind"] = h_df.index
        h_df["avg_diff"] = h_df["full_avg"] - h_df["h_avg"]
        corr_results, data = self.compute_subject_corr(h_df, ["ind", "avg_diff"])
        corr_results.to_csv(
            self.get_output_path() / "rq2_size_heurisitc_corr.csv", index=False
        )

    def generate_outliers_stats(self):
        sids = []
        outlier_count = []
        for subject, sid in self.subject_id_map.items():
            outliers_df = pd.read_csv(
                self.config.data_path
                / subject
                / "tsp_accuracy_results"
                / "full-outliers"
                / "outliers.csv"
            )
            sids.append(sid)
            outlier_count.append(len(outliers_df))

        outlier_count_df = pd.DataFrame(
            {"S_ID": sids, "outliers": outlier_count}
        ).sort_values("outliers", ascending=False, ignore_index=True)
        outlier_count_df.to_csv(
            self.get_output_path() / "rq2_outlier_count.csv", index=False
        )

        def get_all_and_outliers_results(subjects):
            a_builds = []
            a_apfdcs = []
            o_builds = []
            o_apfdcs = []
            for subject, sid in self.subject_id_map.items():
                if sid not in subjects:
                    continue
                res = pd.read_csv(
                    self.config.data_path
                    / subject
                    / "tsp_accuracy_results"
                    / "full"
                    / "results.csv"
                )
                a_builds.extend(res["build"].values.tolist())
                a_apfdcs.extend(res["apfdc"].values.tolist())
                out_res = pd.read_csv(
                    self.config.data_path
                    / subject
                    / "tsp_accuracy_results"
                    / "full-outliers"
                    / "results.csv"
                )
                o_builds.extend(out_res["build"].values.tolist())
                o_apfdcs.extend(out_res["apfdc"].values.tolist())
            a_df = pd.DataFrame({"build": a_builds, "apfdc": a_apfdcs})
            o_df = pd.DataFrame({"build": o_builds, "apfdc": o_apfdcs})
            return a_df.merge(o_df, on="build", suffixes=["_a", "_o"])

        a_res = get_all_and_outliers_results(outlier_count_df["S_ID"].values.tolist())
        z, a_p = wilcoxon(a_res["apfdc_a"], a_res["apfdc_o"])
        a_cl = pg.compute_effsize(
            a_res["apfdc_a"], a_res["apfdc_o"], paired=True, eftype="CLES"
        )

        outlier_subjects = outlier_count_df[outlier_count_df["outliers"] > 0][
            "S_ID"
        ].values.tolist()
        o_res = get_all_and_outliers_results(outlier_subjects)
        z, o_p = wilcoxon(o_res["apfdc_a"], o_res["apfdc_o"])
        o_cl = pg.compute_effsize(
            o_res["apfdc_a"], o_res["apfdc_o"], paired=True, eftype="CLES"
        )

        comp_df = pd.DataFrame(
            {
                "name": ["AllProjects", "OutlierProjects"],
                "a_avg": [a_res["apfdc_a"].mean(), o_res["apfdc_a"].mean()],
                "o_avg": [a_res["apfdc_o"].mean(), o_res["apfdc_o"].mean()],
                "p-value": [a_p, o_p],
                "cl": [a_cl, o_cl],
            }
        )
        comp_df.to_csv(
            self.get_output_path() / "rq2_outlier_ml_comparison.csv", index=False
        )

    def compute_rankers_results(self, rankers, metric="apfdc"):
        avg_results = {}
        results = {}
        for subject, sid in self.subject_id_map.items():
            avg_results.setdefault("S_ID", []).append(sid)
            res_path = self.config.data_path / subject / "tcp_rankers"
            exp_paths = [res_path / r for r in rankers]
            for exp_path in exp_paths:
                exp = exp_path.name
                eval_df = pd.read_csv(exp_path / "results.csv").sort_values(
                    "build", ignore_index=True
                )
                avg_results.setdefault(exp, []).append(eval_df[metric].mean())
                avg_results.setdefault(f"{exp}-std", []).append(eval_df[metric].std())
                results.setdefault(exp, []).extend(eval_df[metric].values.tolist())
        return pd.DataFrame(avg_results), pd.DataFrame(results)

    def run_friedman_nemenyi_tests(self, m_df, m_cols):
        measurements = []
        for c in m_cols:
            measurements.append(m_df[c].values.tolist())
        stat, p = friedmanchisquare(*measurements)
        alpha = 0.05
        with (self.get_output_path() / "rq2_ranker_friedman.txt").open("w") as f:
            print(
                "Number of Samples=%d, Degrees of Freedom=%d"
                % (len(m_df), len(m_cols) - 1),
                file=f,
            )
            print("Statistic=%.3f, p-value=%.3f" % (stat, p), file=f)
            if p >= alpha:
                print(
                    "No statistically significant difference (fail to reject H0)",
                    file=f,
                )
            else:
                print(
                    "There is at least one statistically significant difference (reject H0)",
                    file=f,
                )

    def generate_rankers_avg_table(self):
        rankers = [
            "RandomForest",
            "MART",
            "ListNet",
            "RankBoost",
            "LambdaMART",
            "CoordinateAscent",
        ]
        rankers_avg, rankers_results = self.compute_rankers_results(rankers)
        rankers_avg = rankers_avg.sort_values(
            "MART", ascending=False, ignore_index=True
        )
        rankers_avg["$S_{ID}$"] = rankers_avg["S_ID"].apply(lambda id: f"$S_{{{id}}}$")
        for ranker in rankers:
            rankers_avg[ranker] = rankers_avg.apply(
                lambda r: float("{:.2f}".format(r[ranker])),
                axis=1,
            )
        rankers_avg = rankers_avg[["$S_{ID}$"] + rankers]
        best_rankers = rankers_avg[rankers].idxmax(axis=1)
        for ranker in rankers:
            rankers_avg[ranker] = rankers_avg[ranker].astype(str)
        for i in rankers_avg.index:
            best_apfdc = rankers_avg.at[i, best_rankers[i]]
            rankers_avg.at[i, best_rankers[i]] = "\\textbf{{{:.2f}}}".format(
                float(best_apfdc)
            )
        with (self.get_output_path() / "rq2_ranker_avg.tex").open("w") as f:
            f.write(rankers_avg.to_latex(index=False, escape=False))

        self.run_friedman_nemenyi_tests(rankers_results, rankers)
        rankers_results.to_csv(
            self.get_output_path() / "rq2_ranker_res.csv", index=False
        )
