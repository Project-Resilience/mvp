import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import constants
from data.eluc_data import ELUCData
from data.torch_data import TorchDataset
from prescriptors.nsga2.torch_prescriptor import Candidate

def supervised_backprop(save_path: Path, ds: TorchDataset):
    """
    Performs supervised backpropagation on the given dataset to create a Candidate.
    """
    train_ds, val_ds = random_split(ds, [int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8)])
    train_dl = DataLoader(train_ds, batch_size=4096, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4096, shuffle=True)

    seed = Candidate(in_size=len(constants.CAO_MAPPING["context"]),
                     hidden_size=16,
                     out_size=len(constants.RECO_COLS))
    optimizer = torch.optim.Adam(seed.model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()

    pbar = tqdm(range(300))
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

def seed_no_change(seed_dir: Path, df: pd.DataFrame, encoded_df: pd.DataFrame):
    """
    Creates a seed that attempts to prescribe the same reco cols as the input.
    """
    ds = TorchDataset(encoded_df[constants.CAO_MAPPING["context"]].to_numpy(), df[constants.RECO_COLS].to_numpy())
    seed_dir.mkdir(parents=True, exist_ok=True)
    supervised_backprop(seed_dir / "no_change.pt", ds)
    
def seed_max_change(seed_dir: Path, df: pd.DataFrame, encoded_df: pd.DataFrame):
    """
    Creates a seed that attempts to prescribe the max change to secdf.
    """
    # Create max change labels
    max_change_recos = df[constants.RECO_COLS].copy()
    reco_use = max_change_recos[constants.RECO_COLS].sum(axis=1)
    max_change_recos[constants.RECO_COLS] = 0
    max_change_recos["secdf"] = reco_use

    ds = TorchDataset(encoded_df[constants.CAO_MAPPING["context"]].to_numpy(), max_change_recos[constants.RECO_COLS].to_numpy())
    
    seed_dir.mkdir(parents=True, exist_ok=True)
    supervised_backprop(seed_dir / "max_change.pt", ds)

if __name__ == "__main__":
    dataset = ELUCData()
    train_df = dataset.train_df.sample(10000)
    encoded_train_df = dataset.get_encoded_train().loc[train_df.index]
    seed_no_change(Path("prescriptors/nsga2/seeds/test"), train_df, encoded_train_df)
    seed_max_change(Path("prescriptors/nsga2/seeds/test"), train_df, encoded_train_df)