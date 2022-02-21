import json
import sys
import pandas as pd
from .ranklib_learner import RankLibLearner
from .services.data_service import DataService
import logging


class HypParamOpt:
    def __init__(self, config):
        self.config = config
        self.hyp_combinations = json.load(open("assets/hyp-3-way-cov.json"))
        self.hyp_names = ["rtype", "srate", "bag", "frate", "tree", "leaf", "shrinkage"]
        self.hyp_values = {
            "rtype": [6, 0],
            "srate": [0.5, 1.0],
            "bag": [150, 600, 300],
            "frate": [0.15, 0.6, 0.3],
            "tree": [5, 3, 1],
            "leaf": [50, 200, 100],
            "shrinkage": [0.05, 0.2, 0.1],
        }

    def run_optimization(self, build_ds_path, hyp_comb_i):
        ds_df = self.prepare_dataset()
        learner = RankLibLearner(self.config)
        results_path = self.config.output_path / "hyp_param_opt"
        ranklib_ds = learner.convert_to_ranklib_dataset(ds_df)
        learner.create_ranklib_training_sets(
            ranklib_ds, results_path, custom_test_builds=[self.config.build]
        )

        params = {}
        combination = self.hyp_combinations[hyp_comb_i]
        for i, param_name in enumerate(self.hyp_names):
            param_value = self.hyp_values[param_name][combination[i]]
            params[param_name] = param_value
        logging.info(
            f"Running build {build_ds_path.name} with hyp-comb-index {hyp_comb_i} {combination}"
        )
        logging.info(params)
        if params["rtype"] == 6:
            params["metric2T"] = "NDCG@10"
            params["metric2t"] = "NDCG@10"
        _, apfdc = learner.train_and_test(
            build_ds_path,
            self.config.best_ranker,
            params,
            ds_df,
            suffix=hyp_comb_i,
        )
        logging.info(f"APFDc {apfdc}")
        apfdc_path = build_ds_path / "apfdc"
        apfdc_path.mkdir(parents=True, exist_ok=True)
        with open(str(apfdc_path / f"apfdc{hyp_comb_i}.txt"), "w") as f:
            f.write(str(apfdc))
        logging.info(f"Done run_optimization")

    def prepare_dataset(self):
        dataset_path = self.config.output_path / "dataset.csv"
        if not dataset_path.exists():
            logging.error("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        dataset_df = pd.read_csv(dataset_path)
        outliers_dataset_df = DataService.remove_outlier_tests(
            self.config.output_path, dataset_df
        )
        return outliers_dataset_df
