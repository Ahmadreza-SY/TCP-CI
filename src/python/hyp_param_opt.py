import optuna
import sys
import pandas as pd
from .ranklib_learner import RankLibLearner
from .feature_extractor.feature import Feature
from .services.data_service import DataService


class HypParamOpt:
    def __init__(self, config):
        self.config = config

    def run_optimization(self, build_ds_path):
        ds_df = self.prepare_dataset()
        learner = RankLibLearner(self.config)
        results_path = self.config.output_path / "hyp_param_opt"
        storage_path = self.config.output_path / "hyp_param_opt" / "storage"
        storage_path.mkdir(exist_ok=True, parents=True)
        ranklib_ds = learner.convert_to_ranklib_dataset(ds_df)
        learner.create_ranklib_training_sets(
            ranklib_ds, results_path, custom_test_builds=[self.config.build]
        )
        obj = self.create_objective(build_ds_path, learner, ds_df)
        study_name = f"{self.config.output_path.name}-{build_ds_path.name}"
        storage_name = f"mysql://{self.config.mysql_credentials}@localhost/hypopt"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
        )
        study.optimize(obj, n_trials=self.config.n_trials)

    def prepare_dataset(self):
        dataset_path = self.config.output_path / "dataset.csv"
        if not dataset_path.exists():
            print("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()
        dataset_df = pd.read_csv(dataset_path)
        outliers_dataset_df = DataService.remove_outlier_tests(
            self.config.output_path, dataset_df
        )
        return outliers_dataset_df

    def create_objective(self, build_ds_path, learner, ds_df):
        def objective(trial):
            rtype = trial.suggest_categorical("rtype", [6, 0])
            srate = trial.suggest_categorical("srate", [0.5, 1.0])
            bag = trial.suggest_categorical("bag", [150, 600, 300])
            frate = trial.suggest_categorical("frate", [0.15, 0.6, 0.3])
            tree = trial.suggest_categorical("tree", [5, 3, 1])
            leaf = trial.suggest_categorical("leaf", [50, 200, 100])
            shrinkage = trial.suggest_categorical("shrinkage", [0.05, 0.2, 0.1])
            params = {
                "bag": bag,
                "srate": srate,
                "frate": frate,
                "rtype": rtype,
                "tree": tree,
                "leaf": leaf,
                "shrinkage": shrinkage,
            }
            if rtype == 6:
                params["metric2T"] = "NDCG@10"
                params["metric2t"] = "NDCG@10"
            _, apfdc = learner.train_and_test(
                build_ds_path,
                self.config.best_ranker,
                params,
                ds_df,
                suffix=trial.number,
            )
            return apfdc

        return objective
