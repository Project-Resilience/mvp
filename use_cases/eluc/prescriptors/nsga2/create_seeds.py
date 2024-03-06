"""
Generates PyTorch seeds for the NSGA-II prescriptor.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import constants
from data.eluc_data import ELUCData
from data.torch_data import TorchDataset
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from prescriptors.nsga2.candidate import Candidate
from prescriptors.nsga2.torch_prescriptor import TorchPrescriptor

def supervised_backprop(save_path: Path, ds: TorchDataset, candidate_params: dict, n_epochs=300):
    """
    Performs supervised backpropagation on the given dataset to create a Candidate.
    """
    train_ds, val_ds = random_split(ds, [int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8)])
    train_dl = DataLoader(train_ds, batch_size=4096, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4096, shuffle=True)

    seed = Candidate(**candidate_params)
    optimizer = torch.optim.Adam(seed.model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()

    pbar = tqdm(range(n_epochs))
    for _ in pbar:
        seed.train()
        for X, Y in train_dl:
            optimizer.zero_grad()
            reco_tensor = seed(X)
            loss = loss_fn(reco_tensor, Y)
            loss.backward()
            optimizer.step()

        seed.eval()
        total_loss = 0
        n = 0
        with torch.no_grad():
            for X, Y in val_dl:
                reco_tensor = seed(X)
                loss = loss_fn(reco_tensor, Y)
                total_loss += loss.item() * len(X)
                n += len(X)

        pbar.set_postfix({"val loss": total_loss / n})
    
    torch.save(seed.state_dict(), save_path)

def seed_no_change(seed_dir: Path,
                   df: pd.DataFrame,
                   encoded_df: pd.DataFrame,
                   candidate_params: dict,
                   n_epochs=300):
    """
    Creates a seed that attempts to prescribe the same reco cols as the input.
    """
    ds = TorchDataset(encoded_df[constants.CAO_MAPPING["context"]].to_numpy(),
                      df[constants.RECO_COLS].to_numpy())
    
    seed_dir.mkdir(parents=True, exist_ok=True)
    supervised_backprop(seed_dir / "no_change.pt", ds, candidate_params, n_epochs)
    
def seed_max_change(seed_dir: Path,
                    df: pd.DataFrame,
                    encoded_df: pd.DataFrame,
                    candidate_params: dict,
                    n_epochs=300):
    """
    Creates a seed that attempts to prescribe the max change to secdf.
    Does so by creating artificial labels that are max change to secdf.
    """
    # Create max change labels
    max_change_recos = df[constants.RECO_COLS].copy()
    reco_use = max_change_recos[constants.RECO_COLS].sum(axis=1)
    max_change_recos[constants.RECO_COLS] = 0
    max_change_recos["secdf"] = reco_use

    ds = TorchDataset(encoded_df[constants.CAO_MAPPING["context"]].to_numpy(),
                      max_change_recos[constants.RECO_COLS].to_numpy())

    seed_dir.mkdir(parents=True, exist_ok=True)
    supervised_backprop(seed_dir / "max_change.pt", ds, candidate_params, n_epochs)

def validate_seeds(seed_dir: Path, nn_path: Path, dataset: ELUCData, candidate_params: dict):
    """
    Validates that the seeds' performances match the intended behavior.
    Creates a dummy prescriptor and evaluates the seeds, then prints the results.
    """
    nnp = NeuralNetPredictor()
    nnp.load(nn_path)
    eval_df = dataset.test_df.sample(frac=0.01, random_state=100)
    dummy_prescriptor = TorchPrescriptor(
        2, 1, 0, eval_df, dataset.encoder, nnp, 4096, candidate_params, seed_dir
    )
    candidates = []
    for seed_path in seed_dir.iterdir():
        candidate = Candidate(**candidate_params)
        candidate.load_state_dict(torch.load(seed_path))
        candidates.append(candidate)

    # TODO: We are kinda cheating here but this isn't the intended use of the prescriptor object.
    dummy_prescriptor._evaluate_candidates(candidates)
    for seed_path, candidate in zip(seed_dir.iterdir(), candidates):
        print(f"{seed_path.name} ELUC: {candidate.metrics[0]}, change: {candidate.metrics[1]}")


if __name__ == "__main__":
    # TODO: Add args for candidate params
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
    args = parser.parse_args()

    dataset = ELUCData()
    train_df = dataset.train_df
    if args.n_samples:
        if args.n_samples < 1:
            train_df = train_df.sample(frac=args.n_samples, random_state=100)
        else:
            train_df = train_df.sample(n=int(args.n_samples), random_state=100)
    encoded_train_df = dataset.get_encoded_train().loc[train_df.index]
    candidate_params = {
        "in_size": len(constants.CAO_MAPPING["context"]),
        "hidden_size": 16,
        "out_size": len(constants.RECO_COLS)
    }
    seed_dir = Path(args.seed_dir)
    seed_no_change(seed_dir, train_df, encoded_train_df, candidate_params, args.n_epochs)
    seed_max_change(seed_dir, train_df, encoded_train_df, candidate_params, args.n_epochs)
    if args.validate:
        nn_path = Path(args.nn_path)
        validate_seeds(seed_dir, nn_path, dataset, candidate_params)