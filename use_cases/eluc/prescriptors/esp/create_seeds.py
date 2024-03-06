import argparse
from pathlib import Path
import json

import pandas as pd
import tensorflow as tf
from keras.models import load_model

from data import constants
from data.eluc_data import ELUCData
from prescriptors.esp.unileaf_prescriptor import UnileafPrescriptor
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor

def create_template_model():
    """
    TODO: The architecture is currently hard-coded. Need to figure out how to do this 
    like in PyTorch.
    Creates keras template prescriptor given architecture from paper:
        Input layer for each context variable
        Dense layer for each context variable hidden size 16
        Tanh activation
        Output as reco_land_use vector
    """
    inputs = [tf.keras.Input(shape=(1,), name=f"{col}_input") for col in constants.CAO_MAPPING["context"]]
    dense = [tf.keras.layers.Dense(16, name=constants.CAO_MAPPING["context"][i])(inputs[i]) for i in range(len(inputs))]
    add4 = tf.keras.layers.Add()(dense)
    activation = tf.keras.layers.Activation("tanh", name="first_hidden_activation")(add4)
    output = tf.keras.layers.Dense(len(constants.RECO_COLS), name="reco_land_use")(activation)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def seed_no_change(seed_dir: Path, df: pd.DataFrame, encoded_df: pd.DataFrame, n_epochs=300):
    """
    Creates seed model that attempts to prescribe zero change.
    This is now feasible because we no longer softmax the output but instead linearly scale them.
    """

    no_change_preds = df[constants.RECO_COLS].copy()
    y_train = no_change_preds.to_numpy()
    X_train = [encoded_df[col].values for col in constants.CAO_MAPPING["context"]]

    no_change_model = create_template_model()
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    no_change_model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mae'])
    no_change_model.fit(X_train, y_train, epochs=n_epochs, batch_size=4096, verbose=1)

    seed_dir.mkdir(parents=True, exist_ok=True)
    no_change_model.save(seed_dir / "1_1.h5")

def seed_max_change(seed_dir: Path, df: pd.DataFrame, encoded_df: pd.DataFrame, n_epochs=300, best_col="secdf"):
    """
    Creates seed model that attempts to prescribe maximum change.
    Moves all possible land use to best_col which is secdf by default.
    """
    # Move all the land use to secdf
    land_use = df[constants.RECO_COLS].sum(axis=1)
    max_change_preds = df[constants.RECO_COLS].copy()
    max_change_preds[constants.RECO_COLS] = 0
    max_change_preds[best_col] = land_use

    y_train = max_change_preds.to_numpy()
    X_train = [encoded_df[col].values for col in constants.CAO_MAPPING["context"]]
    
    max_change_model = create_template_model()
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    max_change_model.compile(optimizer=opt, loss='mean_absolute_error', metrics=['mae'])
    max_change_model.fit(X_train, y_train, epochs=n_epochs, batch_size=4096, verbose=1)

    seed_dir.mkdir(parents=True, exist_ok=True)
    max_change_model.save(seed_dir / "1_2.h5")

def validate_seeds(seed_dir: Path, nn_path: Path, presc_cfg_path:Path, dataset: ELUCData):
    """
    TODO: This is pretty yucky right now and exposes some internals in the dummy prescriptor,
    will have to play around with the SWE side of the prescriptors to make this work better.
    Validates that the seeds' performances match the intended behavior.
    Creates a dummy prescriptor and evaluates the seeds, then prints the results.
    """
    nnp = NeuralNetPredictor()
    nnp.load(nn_path)
    with open(presc_cfg_path) as f:
        presc_config = json.load(f)
    dummy_prescriptor = UnileafPrescriptor(presc_config,
                                    dataset.train_df.iloc[:1],
                                    dataset.encoder,
                                    [nnp])
    
    test_df = dataset.test_df.sample(frac=0.01, random_state=100)
    context_df = test_df[constants.CAO_MAPPING["context"]]

    for seed_path in seed_dir.iterdir():
        candidate = load_model(seed_path)
        encoded_context_df = dataset.encoder.encode_as_df(context_df)
        reco_land_use = dummy_prescriptor.prescribe(candidate, encoded_context_df)
        reco_df = pd.DataFrame(reco_land_use["reco_land_use"].tolist(), columns=constants.RECO_COLS)
        context_actions_df = dummy_prescriptor._reco_to_context_actions(reco_df, encoded_context_df)
        context_actions_df = context_actions_df.set_index(context_df.index)

        eluc_df, change_df = dummy_prescriptor.predict_metrics(context_actions_df)
        print(f"{seed_path.name} ELUC: {eluc_df['ELUC'].mean()}, change: {change_df['change'].mean()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_dir", type=str, help="Directory to save seeds to", required=True)
    parser.add_argument("--n_samples", type=float, default=10000,
                        help="How much of the dataset to use for training. \
                            If <1 uses a proportion of the dataset, \
                            otherwise uses a flat number.")
    parser.add_argument("--n_epochs", type=int, default=300, help="Number of epochs to train for.")
    parser.add_argument("--validate", default=True, help="Whether to validate the seeds after training.")
    parser.add_argument("--nn_path", type=str, default="predictors/neural_network/trained_models/no_overlap_nn",
                        help="Path to saved neural network model.")
    parser.add_argument("--presc_cfg_path", type=str, default="prescriptors/esp/unileaf_configs/config-loctime-crop-nosoft.json",
                        help="Path to prescriptor configuration.")
    args = parser.parse_args()

    dataset = ELUCData()

    # Take small subset for training, we really don't need more and just need the model to converge
    train_df = dataset.train_df
    if args.n_samples:
        if args.n_samples < 1:
            train_df = train_df.sample(frac=args.n_samples, random_state=100)
        else:
            train_df = train_df.sample(n=int(args.n_samples), random_state=100)
    encoded_train_df = dataset.get_encoded_train().loc[train_df.index]

    seed_dir = Path(args.seed_dir)

    seed_no_change(seed_dir, train_df, encoded_train_df, args.n_epochs)
    seed_max_change(seed_dir, train_df, encoded_train_df, args.n_epochs)

    if args.validate:
        nn_path = Path(args.nn_path)
        presc_cfg_path = Path(args.presc_cfg_path)
        validate_seeds(seed_dir, nn_path, presc_cfg_path, dataset)

    