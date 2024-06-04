"""
Base implementation of the land use prescriptor as used in the paper.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import constants
from data.eluc_data import ELUCEncoder
from data.torch_data import TorchDataset
from prescriptors.nsga2.candidate import Candidate
from prescriptors.prescriptor import Prescriptor

class LandUsePrescriptor(Prescriptor):
    """
    Prescriptor object that wraps around a single candidate that was trained via.
    evolution using NSGA-II.
    """
    def __init__(self, candidate: Candidate, encoder: ELUCEncoder, batch_size: int=4096):
        self.candidate = candidate
        self.encoder = encoder
        self.batch_size = batch_size

    def _reco_tensor_to_df(self, reco_tensor: torch.Tensor, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts raw Candidate neural network output tensor to scaled dataframe.
        Sets the indices of the recommendations so that we can subtract from the context to get
        the land diffs.
        """
        reco_df = pd.DataFrame(reco_tensor.cpu().numpy(), index=context_df.index, columns=constants.RECO_COLS)
        reco_df = reco_df.clip(0, None) # ReLU
        reco_df[reco_df.sum(axis=1) == 0] = 1 # Rows of all 0s are set to 1s
        reco_df = reco_df.div(reco_df.sum(axis=1), axis=0) # Normalize to sum to 1
        reco_df = reco_df.mul(context_df[constants.RECO_COLS].sum(axis=1), axis=0) # Rescale to match original sum
        return reco_df

    def _reco_to_context_actions(self, reco_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts recommendation df and original context df to context + actions df.
        Uses original context to compute diffs based on recommendations - original context.
        """
        assert reco_df.index.isin(context_df.index).all(), "Recommendation index must be a subset of context index."
        presc_actions_df = reco_df - context_df[constants.RECO_COLS]
        presc_actions_df = presc_actions_df.rename(constants.RECO_MAP, axis=1)
        presc_actions_df[constants.NO_CHANGE_COLS] = 0
        context_actions_df = pd.concat([context_df[constants.CAO_MAPPING["context"]],
                                            presc_actions_df[constants.CAO_MAPPING["actions"]]],
                                            axis=1)
        return context_actions_df

    def prescribe(self, context_df) -> pd.DataFrame:
        """
        Prescribes actions from a context.
        Overall flow of prescription:
            1. context_df -> context_tensor
            2. candidate.forward(context_tensor) -> reco_tensor
            3. reco_tensor -> reco_df
            4. context_df, reco_df -> context_actions_df
        """
        # Either create context_dl or used stored one if it exists
        encoded_context_df = self.encoder.encode_as_df(context_df[constants.CAO_MAPPING["context"]])
        encoded_context_ds = TorchDataset(encoded_context_df.to_numpy(),
                                          np.zeros((len(encoded_context_df), len(constants.RECO_COLS))))
        encoded_context_dl = DataLoader(encoded_context_ds, batch_size=self.batch_size, shuffle=False)
        return self.torch_prescribe(context_df, encoded_context_dl)

    def torch_prescribe(self, context_df: pd.DataFrame, encoded_context_dl: DataLoader):
        """
        Prescribes straight from a torch DataLoader so that we can avoid the overhead of converting from pandas
        during evolution.
        """
        # Aggregate recommendations
        reco_list = []
        with torch.no_grad():
            for X, _ in encoded_context_dl:
                X = X.to(self.candidate.device)
                recos = self.candidate(X)
                reco_list.append(recos)
            reco_tensor = torch.concatenate(reco_list, dim=0)

            # Convert recommendations into context + actions
            reco_df = self._reco_tensor_to_df(reco_tensor, context_df)

        context_actions_df = self._reco_to_context_actions(reco_df, context_df)
        return context_actions_df

    def save(self, path: Path):
        """
        Saves the prescriptor to disk.
        """
        path.mkdir(parents=True, exist_ok=True)
        cand_params = {
            "in_size": self.candidate.in_size,
            "hidden_size": self.candidate.hidden_size,
            "out_size": self.candidate.out_size
        }
        with open(path / "cand_params.json", "w", encoding="utf-8") as file:
            json.dump(cand_params, file)
        self.encoder.save_fields(path / "fields.json")
        torch.save(self.candidate.state_dict(), path / "model.pt")

    @classmethod
    def load(cls, path: Path) -> "LandUsePrescriptor":
        """
        Loads a prescriptor from disk.
        """
        with open(path / "cand_params.json", "r", encoding="utf-8") as file:
            cand_params = json.load(file)
        candidate = Candidate(**cand_params)
        candidate.load_state_dict(torch.load(path / "model.pt"))
        encoder = ELUCEncoder.from_json(path / "fields.json")
        return cls(candidate, encoder)
