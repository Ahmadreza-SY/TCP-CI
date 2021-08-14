import pandas as pd
from scipy.stats import wilcoxon
import pingouin as pg
import matplotlib.pyplot as plt
from .rq1_resutls_analyzer import RQ1ResultAnalyzer
from ..feature_extractor.feature import Feature
import seaborn as sns


class RQ2ResultAnalyzer:
    def __init__(self, config, subject_id_map):
        self.config = config
        self.subject_id_map = subject_id_map

    def analyze_results(self):
        self.generate_accuracy_avg_tables()
        self.generate_feature_group_acc_test_tables()
        self.generate_feature_usage_freq_table()
        self.generate_tc_age_histogram()

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
        results = results.sort_values("full", ascending=False, ignore_index=True)
        results["S_{ID}"] = results["S_ID"].apply(lambda id: f"$S_{{{id}}}$")
        results["Full-inc"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(r["full"], r["full-std"]), axis=1
        )
        results["Impacted-ex"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                r["wo-impacted"], r["wo-impacted-std"]
            ),
            axis=1,
        )
        results["\\texit{TES\\_M}"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(r["W-Code"], r["W-Code-std"]),
            axis=1,
        )
        results["\\texit{REC\\_M}"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                r["W-Execution"], r["W-Execution-std"]
            ),
            axis=1,
        )
        results["\\texit{COV\\_M}"] = results.apply(
            lambda r: "${:.2f} {{\pm}} {:.2f}$".format(
                r["W-Coverage"], r["W-Coverage-std"]
            ),
            axis=1,
        )
        return results[
            [
                "S_{ID}",
                "Full-inc",
                "Impacted-ex",
                "\\texit{TES\\_M}",
                "\\texit{REC\\_M}",
                "\\texit{COV\\_M}",
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
        samples = {}
        for subject, _ in self.subject_id_map.items():
            results_df = pd.read_csv(
                self.config.data_path
                / subject
                / "tsp_accuracy_results"
                / "full"
                / "results.csv"
            ).sort_values("build", ignore_index=True)
            samples.setdefault("build", []).extend(results_df["build"].values.tolist())
            samples.setdefault("full", []).extend(results_df[metric].values.tolist())
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
            x, y = samples_df["full"].values, samples_df[exp].values
            z, p = wilcoxon(x, y)
            wilcoxon_res["A"].append("full")
            wilcoxon_res["B"].append(exp)
            wilcoxon_res["p-value"].append(p)
            cl = pg.compute_effsize(x, y, paired=True, eftype="CLES")
            wilcoxon_res["CL"].append(cl)
        return pd.DataFrame(wilcoxon_res).sort_values("p-value", ignore_index=True)

    def generate_feature_group_acc_test_table(self, experiments, name):
        apfd = self.run_feature_group_acc_tests("apfd", experiments)
        apfdc = self.run_feature_group_acc_tests("apfdc", experiments)
        apfd.to_csv(self.get_output_path() / f"rq2_apfd_{name}_tests.csv", index=False)
        apfdc.to_csv(
            self.get_output_path() / f"rq2_apfdc_{name}_tests.csv", index=False
        )

        def format_columns(df):
            name_dict = {
                "W-Code": "TES_M",
                "W-Execution": "REC_M",
                "W-Coverage": "COV_M",
            }
            df["B"] = df["B"].apply(lambda v: name_dict[v] if v in name_dict else v)
            df["B"] = df["B"].apply(lambda v: v.replace("_", "\\_").replace("wo-", ""))
            df["A"] = df["A"].apply(lambda v: v.title())
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
        tests_df = apfd.merge(apfdc, on=["A", "B"], suffixes=["-apfd", "-apfdc"])
        with (self.get_output_path() / f"rq2_{name}_tests.tex").open("w") as f:
            f.write(tests_df.to_latex(index=False, escape=False))

    def generate_feature_group_acc_test_tables(self):
        without_experiments = [f"wo-{fg}" for fg in RQ1ResultAnalyzer.FEATURE_GROUPS]
        self.generate_feature_group_acc_test_table(without_experiments, "without")

        with_experiments = ["W-Code", "W-Execution", "W-Coverage"]
        self.generate_feature_group_acc_test_table(with_experiments, "with")

    def compute_avg_feature_usage_freq(self):
        freq_samples = {}
        for subject, _ in self.subject_id_map.items():
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
                    self.config.data_path / subject / "tsp_accuracy_results" / "full"
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
        avg_usage_freq = avg_usage_freq[:10]
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
        for subject, _ in self.subject_id_map.items():
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

        fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=100)
        ax.set(xlabel="Normalized Age", ylabel="Failure Count")
        sns.histplot(x, ax=ax, kde=False, bins=10, color="blue")
        plt.savefig(self.get_output_path() / "rq2_tc_age_hist.png", bbox_inches="tight")
