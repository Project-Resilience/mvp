"""
Generates PyTorch seeds for the NSGA-II prescriptor.
"""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from prsdk.data.torch_data import TorchDataset
from prsdk.persistence.serializers.sklearn_serializer import SKLearnSerializer

from data import constants
from data.eluc_data import ELUCData
from prescriptors.heuristics.heuristics import PerfectHeuristic
from prescriptors.nsga2.candidate import Candidate


def supervised_backprop(epochs: int, save_path: Path, ds: TorchDataset):
    """
    Performs supervised backpropagation on the given dataset to create a Candidate.
    """
    train_ds, val_ds = random_split(ds, [int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8)])
    train_dl = DataLoader(train_ds, batch_size=4096, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False)

    seed = Candidate(in_size=len(constants.CAO_MAPPING["context"]),
                     hidden_size=16,
                     out_size=len(constants.RECO_COLS))
    optimizer = torch.optim.AdamW(seed.model.parameters())
    loss_fn = torch.nn.MSELoss()

    pbar = tqdm(range(epochs))
    for _ in pbar:
        seed.train()
        for X, Y in train_dl:
            X, Y = X.float(), Y.float()
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
                X, Y = X.float(), Y.float()
                reco_tensor = seed(X)
                loss = loss_fn(reco_tensor, Y)
                total_loss += loss.item() * len(X)
                n += len(X)

        pbar.set_postfix({"val loss": total_loss / n})

    torch.save(seed.state_dict(), save_path)


def seed_no_change(epochs: int, seed_dir: Path, df: pd.DataFrame, encoded_df: pd.DataFrame):
    """
    Creates a seed that attempts to prescribe the same reco cols as the input.
    """
    ds = TorchDataset(encoded_df[constants.CAO_MAPPING["context"]].to_numpy(), df[constants.RECO_COLS].to_numpy())
    seed_dir.mkdir(parents=True, exist_ok=True)
    supervised_backprop(epochs, seed_dir / "no_change.pt", ds)


def seed_max_change(epochs: int, seed_dir: Path, df: pd.DataFrame, encoded_df: pd.DataFrame):
    """
    Creates a seed that attempts to prescribe the max change to secdf.
    Does so by creating artificial labels that are max change to secdf.
    """
    # Create max change labels
    max_change_recos = df[constants.RECO_COLS].copy()
    reco_use = max_change_recos[constants.RECO_COLS].sum(axis=1)
    max_change_recos[constants.RECO_COLS] = 0
    max_change_recos["secdf"] = reco_use

    encoded_context_np = encoded_df[constants.CAO_MAPPING["context"]].to_numpy()
    max_change_recos_np = max_change_recos[constants.RECO_COLS].to_numpy()
    ds = TorchDataset(encoded_context_np, max_change_recos_np)

    seed_dir.mkdir(parents=True, exist_ok=True)
    supervised_backprop(epochs, seed_dir / "max_change.pt", ds)


def seed_no_crop_change(epochs: int, seed_dir: Path, df: pd.DataFrame, encoded_df: pd.DataFrame):
    """
    Creates a seed that attempts to prescribe no change to crops while still maximizing forest.
    """
    # Create max change labels
    max_change_recos = df[constants.RECO_COLS].copy()
    no_crop_recos = [col for col in constants.RECO_COLS if "crop" not in col]
    reco_use = max_change_recos[no_crop_recos].sum(axis=1)
    max_change_recos[no_crop_recos] = 0
    max_change_recos["secdf"] = reco_use

    encoded_context_np = encoded_df[constants.CAO_MAPPING["context"]].to_numpy()
    max_change_recos_np = max_change_recos[constants.RECO_COLS].to_numpy()
    ds = TorchDataset(encoded_context_np, max_change_recos_np)

    seed_dir.mkdir(parents=True, exist_ok=True)
    supervised_backprop(epochs, seed_dir / "no_crop_change.pt", ds)


def seed_perfect(epochs: int, seed_dir: Path, df: pd.DataFrame, encoded_df: pd.DataFrame, heuristic: PerfectHeuristic):
    """
    Creates a seed that uses the perfect heuristic at a given percentage to create the labels.
    """
    reco_df = heuristic.reco_heuristic(df[constants.CAO_MAPPING["context"]])
    reco_np = reco_df[constants.RECO_COLS].to_numpy()
    encoded_context_np = encoded_df[constants.CAO_MAPPING["context"]].to_numpy()
    ds = TorchDataset(encoded_context_np, reco_np)

    seed_dir.mkdir(parents=True, exist_ok=True)
    pct_name = str(round(heuristic.pct, 2)).replace(".", "_")
    supervised_backprop(epochs, seed_dir / f"perfect-{pct_name}.pt", ds)


def seed_rhea(seed_dir: str):
    """
    Seeds for the RHEA experiment. Requires the linreg to be trained.
    """
    rhea_seed_dir = Path(seed_dir) if seed_dir else Path("prescriptors/nsga2/seeds/eds-rhea")
    dataset = ELUCData.from_hf()
    train_df = dataset.train_df.sample(frac=0.1, random_state=42)
    encoded_train_df = dataset.get_encoded_train().loc[train_df.index]
    epochs = 15

    # Set up for heuristic
    serializer = SKLearnSerializer()
    linreg = serializer.load("predictors/trained_models/danyoung--eluc-global-linreg")
    coefs = linreg.model.coef_
    coef_dict = dict(zip(constants.LAND_USE_COLS, coefs))
    reco_coefs = []
    for col in constants.RECO_COLS:
        reco_coefs.append(coef_dict[col])
    pcts = list(range(0, 101, 5))
    pcts = [pct / 100 for pct in pcts]

    # Train a seed for each pct value
    for pct in pcts:
        heuristic = PerfectHeuristic(pct, reco_coefs)
        seed_perfect(epochs, rhea_seed_dir, train_df, encoded_train_df, heuristic)


def seed_no_crop(seed_dir: str):
    """
    Seeds for the no crop experiment
    """
    no_crop_seed_dir = Path(seed_dir) if seed_dir else Path("prescriptors/nsga2/seeds/eds-crop")
    dataset = ELUCData.from_hf()
    train_df = dataset.train_df.sample(10000, random_state=42)
    encoded_train_df = dataset.get_encoded_train().loc[train_df.index]
    epochs = 300

    seed_no_change(epochs, no_crop_seed_dir, train_df, encoded_train_df)
    seed_max_change(epochs, no_crop_seed_dir, train_df, encoded_train_df)
    seed_no_crop_change(epochs, no_crop_seed_dir, train_df, encoded_train_df)


def seed_base(seed_dir: str):
    """
    Base seeds for the original experiment
    """
    base_seed_dir = Path(seed_dir) if seed_dir else Path("prescriptors/nsga2/seeds/eds")
    dataset = ELUCData.from_hf()
    train_df = dataset.train_df.sample(frac=0.1, random_state=42)
    encoded_train_df = dataset.get_encoded_train().loc[train_df.index]
    epochs = 15

    seed_no_change(epochs, base_seed_dir, train_df, encoded_train_df)
    seed_max_change(epochs, base_seed_dir, train_df, encoded_train_df)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--type", type=str)
    parser.add_argument("--seed_dir", type=str, default=None)
    args = parser.parse_args()

    if args.type == "rhea":
        seed_rhea(args.seed_dir)
    elif args.type == "no_crop":
        seed_no_crop(args.seed_dir)
    elif args.type == "base":
        seed_base(args.seed_dir)
    elif args.type == "all":
        seed_rhea(None)
        seed_no_crop(None)
        seed_base(None)
    else:
        raise ValueError(f"Unknown seed type {args.type}. Must be one of: rhea, no_crop, base, all")
