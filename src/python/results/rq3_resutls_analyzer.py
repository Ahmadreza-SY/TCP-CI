import pandas as pd
import matplotlib.pyplot as plt


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

    def compute_accuracy_decay(self, subject_path):
        decay_results = []
        decay_paths = [
            p for p in (subject_path / "decay_datasets").glob("*") if p.is_dir()
        ]
        for decay_path in decay_paths:
            decay_result = pd.read_csv(decay_path / "results.csv")
            decay_result["window"] = decay_result.index
            decay_results.append(decay_result)
        if len(decay_results) == 0:
            return pd.DataFrame()
        return pd.concat(decay_results, ignore_index=True)

    def plot_decay_graphs(self, metric):
        fig, axs = plt.subplots(4, 4, figsize=(18, 12))
        fig.suptitle(f"Subject {metric} Decay")
        fig.subplots_adjust(hspace=0.4, wspace=0.25)
        i = 0
        for subject, sid in self.subject_id_map.items():
            subject_path = self.config.data_path / subject
            ax = axs[int(i / 4), i % 4]
            ax.set(xlabel="Window", ylabel=metric)
            decay_results_df = self.compute_accuracy_decay(subject_path)
            decay_results_df = decay_results_df[
                decay_results_df["window"] <= RQ3ResultAnalyzer.WINDOW_THRESHOLD
            ]
            win_avg_df = (
                decay_results_df[["window", metric]]
                .groupby("window", as_index=False)
                .mean()
            )
            ax.plot(win_avg_df["window"].values, win_avg_df[metric].values)
            ax.set_title(f"$S_{{{sid}}}$")
            i += 1
        plt.savefig(
            self.get_output_path() / f"rq3_{metric}_decay.png", bbox_inches="tight"
        )
