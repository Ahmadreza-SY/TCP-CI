import optuna
import sys
import pandas as pd
from .ranklib_learner import RankLibLearner
from .feature_extractor.feature import Feature
from .services.data_service import DataService


class HypParamOpt:
    def __init__(self, config):
        self.config = config

    def run_optimization(self):
        ds_df = self.prepare_dataset()
        learner = RankLibLearner(self.config)
        results_path = self.config.output_path / "hyp_param_opt"
        storage_path = self.config.output_path / "hyp_param_opt" / "storage"
        storage_path.mkdir(exist_ok=True, parents=True)
        ranklib_ds = learner.convert_to_ranklib_dataset(ds_df)
        learner.create_ranklib_training_sets(ranklib_ds, results_path)
        ds_paths = list(p for p in results_path.glob("*") if p.is_dir())
        best_params = []
        for build_ds_path in ds_paths:
            obj = self.create_objective(build_ds_path, learner)
            study_name = f"{self.config.output_path.name}-{build_ds_path.name}"
            storage_name = f"sqlite:///{str(storage_path)}/{study_name}.db"
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,
            )
            # TODO: parameterize n_trials
            study.optimize(obj, n_trials=3)
            best_params.append((build_ds_path.name, study.best_params))
        return best_params

    def prepare_dataset(self):
        dataset_path = self.config.output_path / "dataset.csv"
        if not dataset_path.exists():
            print("No dataset.csv found in the output directory. Aborting ...")
            sys.exit()

        dataset_df = pd.read_csv(dataset_path)
        builds_count = dataset_df[Feature.BUILD].nunique()
        if builds_count <= self.config.test_count:
            print(
                f"Not enough builds for training: require at least {self.config.test_count + 1}, found {builds_count}"
            )
            sys.exit()

        outliers_dataset_df = DataService.remove_outlier_tests(
            self.config.output_path, dataset_df
        )
        return outliers_dataset_df

    def create_objective(self, build_ds_path, learner):
        def objective(trial):
            bag = trial.suggest_categorical("bag", [100, 300, 600, 1000])
            srate = trial.suggest_categorical("srate", [1.0, 0.5])
            frate = trial.suggest_categorical("frate", [0.1, 0.3, 0.6, 1.0])
            rtype = trial.suggest_categorical("rtype", [0, 6])
            tree = trial.suggest_categorical("tree", [1, 10, 30])
            leaf = trial.suggest_categorical("leaf", [50, 100, 200, 500])
            shrinkage = trial.suggest_categorical("shrinkage", [0.01, 0.1, 0.2])
            params = {
                "bag": bag,
                "srate": srate,
                "frate": frate,
                "rtype": rtype,
                "tree": tree,
                "leaf": leaf,
                "shrinkage": shrinkage,
            }
            _, apfdc = learner.train_and_test(
                build_ds_path, self.config.best_ranker, params, suffix=trial.number
            )
            return apfdc

        return objective
