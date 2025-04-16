"""
Script used to generate the heatmap data from our old models.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from prsdk.persistence.serializers.neural_network_serializer import NeuralNetSerializer
from prsdk.persistence.serializers.sklearn_serializer import SKLearnSerializer
from prsdk.predictors.predictor import Predictor
from prsdk.predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from prsdk.predictors.sklearn_predictors.linear_regression_predictor import LinearRegressionPredictor

from data import constants
from data.eluc_data import ELUCData


def create_heatmap_data_from_predictor(model: Predictor, sample: pd.DataFrame):
    """
    Creates a 2d heatmap of the outcomes of the model on synthetic data.
    The columns correspond to current land type and the rows are the land type converted to.
    Synthetic data is created by using the non-land features of the sample and taking the cross
    join between the artificial land use change from 0 to 1 of every type.
    """
    dummy_data = []
    for i in range(len(constants.LAND_USE_COLS)):
        for j in range(len(constants.LAND_USE_COLS)):
            row = [0 for _ in range(len(constants.LAND_USE_COLS) * 2)]
            row[i] = 1.0
            if i == j:
                row[len(constants.LAND_USE_COLS) + j] = 0.0
            else:
                row[len(constants.LAND_USE_COLS) + j] = 1.0
                row[len(constants.LAND_USE_COLS) + i] = -1.0
            dummy_data.append(dict(zip(constants.LAND_USE_COLS + constants.DIFF_LAND_USE_COLS, row)))

    dummy_df = pd.DataFrame(dummy_data)

    # Gets sample of lat/lon/time/cell_area
    non_land_df = sample[constants.NONLAND_FEATURES]
    nn_input_df = dummy_df.merge(non_land_df, how="cross")
    preds = model.predict(nn_input_df)
    # Aggregate samples. Since pandas merge maintains left key order, we can just sum each group of len(samples)
    # Thanks, ChatGPT!
    chunks = np.split(preds, len(preds) // len(sample))
    sums = np.sum(chunks, axis=1)
    preds = sums / len(sample)

    # Rearrange the heatmap data into 2D shape
    heatmap_data = np.zeros((len(constants.LAND_USE_COLS), len(constants.LAND_USE_COLS)))
    for i, pred in enumerate(preds):
        heatmap_data[i // len(constants.LAND_USE_COLS), i % len(constants.LAND_USE_COLS)] = pred

    # Hide the ability to move land to primf/primn
    idxs = [constants.LAND_USE_COLS.index(col) for col in constants.RECO_COLS]
    non_idxs = [constants.LAND_USE_COLS.index(col) for col in constants.LAND_USE_COLS if col not in constants.RECO_COLS]
    heatmap_data = heatmap_data[idxs + non_idxs, :]
    heatmap_data = heatmap_data[:, idxs]
    return heatmap_data


def generate_linreg_nn_data():
    """
    Main experiment logic. Loads the saved nn and linreg models, creates a sample of the test set, and then does
    prediction on the synthetic data created by the sample. Then we save this data to disk.
    """
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

    nnp = nnp_serializer.load(Path("predictors/trained_models/danyoung--eluc-global-nn"))

    linreg_config = {
        "features": constants.DIFF_LAND_USE_COLS,
        "n_jobs": -1,
    }
    linreg = LinearRegressionPredictor(linreg_config)
    sklearn_serializer = SKLearnSerializer()
    linreg = sklearn_serializer.load("predictors/trained_models/danyoung--eluc-global-linreg")

    dataset = ELUCData.from_hf()
    sample = dataset.test_df.sample(frac=0.01, random_state=100)

    nn_heatmap_data = create_heatmap_data_from_predictor(nnp, sample)
    np.save("experiments/heatmap/nn_heatmap_data.npy", nn_heatmap_data)

    linreg_heatmap_data = create_heatmap_data_from_predictor(linreg, sample)
    np.save("experiments/heatmap/linreg_heatmap_data.npy", linreg_heatmap_data)
