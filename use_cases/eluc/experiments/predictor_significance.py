"""
Performs predictor significance experiments.
Trains and tests different models n times and finds the standard error of their MAE.
Times training as well.
"""
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from data.eluc_data import ELUCData
from data import constants
from data.conversion import construct_countries_df
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from predictors.sklearn.sklearn_predictor import RandomForestPredictor, LinearRegressionPredictor

def train_and_test(n: int,
                   model_constructor,
                   config: dict,
                   train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   train_regions: list,
                   save_path: Path,
                   override_start_year=None):
    """
    Trains a model n times on each region and evaluates each model on each region.
    :param n: Number of times to train each model on each region.
    :param model_constructor: A function that returns a model.
    :param config: A dictionary of configuration parameters for each model type.
    :param train_df: The training data.
    :param test_df: The testing data.
    :param save_path: The path to save the results.
    :param override_start_year: If not None, overrides the start year of the test data on the ALL region.
        (This is currently only used for the random forest)
    """
    countries_df = construct_countries_df()
    print(f"Saving to: {save_path}")
    save_dir = save_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all regions
    results = []
    for train_region in train_regions:
        print(f"Training on {train_region}")
        if train_region != "ALL":
            countries = constants.COUNTRY_DICT[train_region]
            idx = countries_df[countries_df["abbrevs"].isin(countries)].index.values
            train_region_df = train_df[train_df["country"].isin(idx)]
        else:
            train_region_df = train_df
            if override_start_year:
                print(f"Overriding start year to: {override_start_year}")
                train_region_df = train_region_df.loc[override_start_year:]

        # n times for each region
        for _ in tqdm(range(n)):
            result_row = {"train": train_region}
            model = model_constructor(**config)
            s = time.time()
            _ = model.fit(train_region_df, train_region_df["ELUC"])
            e = time.time()
            result_row["time"] = e - s
            # Evaluate on each region
            for test_region, countries in constants.COUNTRY_DICT.items():
                if test_region != "ALL":
                    idx = countries_df[countries_df["abbrevs"].isin(countries)].index.values
                    test_region_df = test_df[test_df["country"].isin(idx)]
                else:
                    test_region_df = test_df

                mae = mean_absolute_error(model.predict(test_region_df), test_region_df["ELUC"])
                result_row[test_region] = mae

            results.append(result_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path)

def main():
    """
    Main function call that performs significance tests.
    Sets up NN, forest, and lr configs and constructors then passes them into
    train_and_test to train and test the models n times.
    """
    print("Loading data...")
    dataset = ELUCData()

    nn_config = {
        "features": constants.NN_FEATS,
        "label": "ELUC",
        "hidden_sizes": [4096],
        "linear_skip": True,
        "dropout": 0,
        "device": "cuda",
        "epochs": 3,
        "batch_size": 2048,
        "train_pct": 1,
        "step_lr_params": {"step_size": 1, "gamma": 0.1},
    }
    forest_config = {
        "features": constants.NN_FEATS,
        "n_jobs": -1,
        "max_features": "sqrt"
    }
    linreg_config = {
        "features": constants.DIFF_LAND_USE_COLS,
        "n_jobs": -1,
    }
    model_constructors = [NeuralNetPredictor, RandomForestPredictor, LinearRegressionPredictor]
    configs = [nn_config, forest_config, linreg_config]
    model_names = ["neural_network", "random_forest", "linear_regression"]
    train_regions = list(constants.COUNTRY_DICT.keys())
    significance_path = Path("experiments/predictor_significance/no_overlap_fixed")
    for model_constructor, config, model_name in zip(model_constructors, configs, model_names):
        override_start_year = None if model_name != "random_forest" else 1982
        print(model_name)
        train_and_test(30,
                    model_constructor,
                    config,
                    dataset.train_df,
                    dataset.test_df,
                    train_regions,
                    significance_path / f"{model_name}_eval.csv",
                    override_start_year=override_start_year)

if __name__ == "__main__":
    main()
