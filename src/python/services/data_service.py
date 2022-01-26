from ..feature_extractor.feature import Feature
import pandas as pd


class DataService:
    @staticmethod
    def remove_outlier_tests(output_path, dataset_df):
        test_f = dataset_df[dataset_df[Feature.VERDICT] > 0][
            [Feature.TEST, Feature.BUILD]
        ]
        test_fcount = (
            test_f.groupby(Feature.TEST, as_index=False)
            .count()
            .sort_values(Feature.BUILD, ascending=False, ignore_index=True)
        )
        test_fcount["rate"] = (
            test_fcount[Feature.BUILD] / dataset_df[Feature.BUILD].nunique()
        )
        mean, std = test_fcount["rate"].mean(), test_fcount["rate"].std()
        outliers = []
        for _, r in test_fcount.iterrows():
            if abs(r["rate"] - mean) > 3 * std:
                outliers.append(int(r[Feature.TEST]))

        outliers_path = (
            output_path / "tsp_accuracy_results" / "full-outliers" / "outliers.csv"
        )
        outliers_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"test": outliers}).to_csv(outliers_path, index=False)
        result_df = dataset_df[~dataset_df[Feature.TEST].isin(outliers)]
        failed_builds = (
            result_df[result_df[Feature.VERDICT] > 0][Feature.BUILD].unique().tolist()
        )
        return (
            result_df[result_df[Feature.BUILD].isin(failed_builds)]
            .copy()
            .reset_index(drop=True)
        )