"""
Script used to recreate the predictors from the paper. Also can be used to load predictors from HuggingFace.
"""
from pathlib import Path

from sklearn.metrics import mean_absolute_error

from prsdk.persistence.serializers.neural_network_serializer import NeuralNetSerializer
from prsdk.persistence.serializers.sklearn_serializer import SKLearnSerializer
from prsdk.persistence.persistors.hf_persistor import HuggingFacePersistor
from prsdk.predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from prsdk.predictors.sklearn_predictors.linear_regression_predictor import LinearRegressionPredictor
from prsdk.predictors.sklearn_predictors.random_forest_predictor import RandomForestPredictor

from data.eluc_data import ELUCData
from data import constants


def train_nn(save_path: str, dataset: ELUCData):
    """
    Trains the neural network model.
    """
    save_path = Path(save_path) if save_path else Path("predictors/neural_network/trained_models/no_overlap_nn")
    save_path.mkdir(parents=True, exist_ok=True)

    nn_config = {
        "features": constants.NN_FEATS,
        "label": "ELUC",
        "hidden_sizes": [4096],
        "linear_skip": True,
        "dropout": 0,
        "device": "mps",
        "epochs": 3,
        "batch_size": 2048,
        "train_pct": 1,
        "step_lr_params": {"step_size": 1, "gamma": 0.1},
    }
    nnp = NeuralNetPredictor(nn_config)
    nnp_serializer = NeuralNetSerializer()

    _ = nnp.fit(dataset.train_df[nn_config["features"]], dataset.train_df[nn_config["label"]], verbose=True)
    nnp_serializer.save(nnp, save_path)

    preds = nnp.predict(dataset.test_df[nn_config["features"]])
    print(f"MAE Neural Network: {mean_absolute_error(dataset.test_df[nn_config['label']], preds)}")


def train_linreg(save_path: str, dataset: ELUCData):
    """
    Trains the simple linear regression model.
    """
    save_path = Path(save_path) if save_path else Path("predictors/sklearn/trained_models/no_overlap_linreg")
    save_path.mkdir(parents=True, exist_ok=True)

    linreg_config = {
        "features": constants.DIFF_LAND_USE_COLS,
        "n_jobs": -1,
    }
    linreg = LinearRegressionPredictor(linreg_config)

    linreg.fit(dataset.train_df[constants.DIFF_LAND_USE_COLS], dataset.train_df["ELUC"])
    sklearn_serializer = SKLearnSerializer()
    sklearn_serializer.save(linreg, save_path)

    preds = linreg.predict(dataset.test_df[constants.DIFF_LAND_USE_COLS])
    print(f"MAE Linear Regression: {mean_absolute_error(dataset.test_df['ELUC'], preds)}")


def train_rf(save_path: str, dataset: ELUCData):
    """
    Trains the random forest. NOTE: This creates a massive model
    """
    save_path = Path(save_path) if save_path else Path("predictors/sklearn/trained_models/no_overlap_rf")
    save_path.mkdir(parents=True, exist_ok=True)

    forest_config = {
        "features": constants.NN_FEATS,
        "n_jobs": -1,
        "max_features": "sqrt",
        "random_state": 42
    }
    forest = RandomForestPredictor(forest_config)

    forest_year = 1982
    forest.fit(dataset.train_df.loc[forest_year:][constants.NN_FEATS], dataset.train_df.loc[forest_year:]["ELUC"])
    sklearn_serializer = SKLearnSerializer()
    sklearn_serializer.save(forest, Path("predictors/trained_models/eds_rf"))

    preds = forest.predict(dataset.test_df[constants.NN_FEATS])
    print(f"MAE Random Forest: {mean_absolute_error(dataset.test_df['ELUC'], preds)}")


def train_all():
    """
    Main logic running all 3 training runs.
    """
    dataset = ELUCData.from_hf()
    train_nn("predictors/neural_network/trained_models/no_overlap_nn", dataset)
    train_linreg("predictors/sklearn/trained_models/no_overlap_linreg", dataset)
    train_rf("predictors/sklearn/trained_models/no_overlap_rf", dataset)


def load_all():
    """
    Loads all 3 models from HuggingFace.
    """
    save_path = Path("predictors/trained_models")
    save_path.mkdir(parents=True, exist_ok=True)

    nn_serializer = NeuralNetSerializer()
    nn_persistor = HuggingFacePersistor(nn_serializer)

    nn_path = "danyoung/eluc-global-nn"
    nn_persistor.from_pretrained(nn_path, local_dir=save_path / nn_path.replace("/", "--"))

    sklearn_serializer = SKLearnSerializer()
    sklearn_persistor = HuggingFacePersistor(sklearn_serializer)

    linreg_path = "danyoung/eluc-global-linreg"
    rf_path = "danyoung/eluc-global-rf"
    sklearn_persistor.from_pretrained(linreg_path, local_dir=save_path / linreg_path.replace("/", "--"))
    sklearn_persistor.from_pretrained(rf_path, local_dir=save_path / rf_path.replace("/", "--"))


if __name__ == "__main__":
    # train_all()
    load_all()
