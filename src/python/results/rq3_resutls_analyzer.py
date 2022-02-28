import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib


class RQ3ResultAnalyzer:
    WINDOW_THRESHOLD = 45

    def __init__(self, config, subject_id_map):
        self.config = config
        self.subject_id_map = subject_id_map

    def analyze_results(self):
        self.plot_decay_graphs("apfd")
        self.plot_decay_graphs("apfdc")

    def get_output_path(self):
        output_path = self.config.output_path / "RQ3"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def compute_accuracy_decay(self):
        decay_results = []

        for subject, sid in self.subject_id_map.items():
            subject_path = self.config.data_path / subject
            decay_paths = [
                p
                for p in (subject_path / "decay_datasets").glob("*")
                if (p.is_dir() and (p / "results.csv").exists())
            ]
            for decay_path in decay_paths:
                decay_result = pd.read_csv(decay_path / "results.csv")
                decay_result["window"] = decay_result.index
                decay_result["subject"] = [subject] * len(decay_result)
                decay_results.append(decay_result)
        return pd.concat(decay_results, ignore_index=True)

    def plot_decay_graphs(self, metric):
        decay_results_df = self.compute_accuracy_decay()
        decay_results_df = decay_results_df[
            decay_results_df["window"] <= RQ3ResultAnalyzer.WINDOW_THRESHOLD
        ]
        win_avg_df = (
            decay_results_df[["window", metric]]
            .groupby("window", as_index=False)
            .mean()
        )
        windows = win_avg_df["window"].values
        accuracies = win_avg_df[metric].values

        font = {"size": 20}
        matplotlib.rc("font", **font)

        plt.figure(figsize=(16, 9))
        plt.plot(windows, accuracies)
        steady_win = 6
        plt.plot(
            [windows[0], windows[steady_win]],
            [accuracies[0], accuracies[steady_win]],
            "r--",
        )
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().xaxis.set_tick_params(labelbottom=True)
        plt.gca().yaxis.set_tick_params(labelbottom=True)

        formatted_metric = metric.upper() if metric == "apfd" else "Avg. $APFD_C$"
        plt.gca().set(xlabel="Retraining Window (RW)", ylabel=formatted_metric)
        plt.savefig(
            self.get_output_path() / f"rq3_{metric}_decay.png",
            bbox_inches="tight",
            dpi=300,
        )

        slope = (accuracies[steady_win] - accuracies[0]) / (
            windows[steady_win] - windows[0]
        )
        with open(
            str(self.get_output_path() / f"rq3_{metric}_decay_slope.csv"), "w"
        ) as f:
            f.write(str(slope))
