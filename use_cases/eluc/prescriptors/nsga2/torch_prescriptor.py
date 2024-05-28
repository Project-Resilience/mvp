"""
LandUse Prescriptor using PyTorch NNs
"""

import numpy as np
import pandas as pd
import torch

from data import constants
from data.eluc_data import ELUCEncoder
from data.torch_data import TorchDataset
from predictors.predictor import Predictor
from prescriptors.prescriptor import Prescriptor
from prescriptors.nsga2.candidate import Candidate

class TorchPrescriptor(Prescriptor):
    """
    Handles prescriptor candidate evolution
    """
    def __init__(self,
                 eval_df: pd.DataFrame,
                 encoder: ELUCEncoder,
                 predictor: Predictor,
                 batch_size: int,
                 candidate_params: dict):

        self.candidate_params = candidate_params

        # Store eval df if needed
        if eval_df is not None:
            self.eval_df = eval_df
            self.encoded_eval_df = encoder.encode_as_df(eval_df)
            # We cache the training context here so that we don't have to repeatedly convert to tensor.
            # We can pass in our own dataframe later for inference.
            context_ds = TorchDataset(self.encoded_eval_df[constants.CAO_MAPPING["context"]].to_numpy(),
                                    np.zeros((len(self.encoded_eval_df), len(constants.RECO_COLS))))
            self.context_dl = torch.utils.data.DataLoader(context_ds, batch_size=batch_size, shuffle=False)

        self.encoder = encoder
        self.batch_size = batch_size
        self.predictor = predictor

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

    def prescribe(self, candidate: Candidate, context_df=None) -> pd.DataFrame:
        """
        Prescribes actions given a candidate and a context.
        If we don't provide a context_df, we use the stored context_dl to avoid overhead. 
        Otherwise, we create a new dataloader from the given context_df.
        Overall flow of prescription:
            1. context_df -> context_tensor
            2. candidate.forward(context_tensor) -> reco_tensor
            3. reco_tensor -> reco_df
            4. context_df, reco_df -> context_actions_df
        """
        # Either create context_dl or used stored one if it exists
        context_dl = None
        if context_df is not None:
            encoded_context_df = self.encoder.encode_as_df(context_df[constants.CAO_MAPPING["context"]])
            context_ds = TorchDataset(encoded_context_df.to_numpy(),
                                      np.zeros((len(encoded_context_df), len(constants.RECO_COLS))))
            context_dl = torch.utils.data.DataLoader(context_ds, batch_size=self.batch_size, shuffle=False)
        elif self.eval_df is not None:
            context_df = self.eval_df
            context_dl = self.context_dl
        else:
            raise ValueError("No context provided and no eval df stored.")

        # Aggregate recommendations
        reco_list = []
        with torch.no_grad():
            for X, _ in context_dl:
                recos = candidate(X)
                reco_list.append(recos)
            reco_tensor = torch.concatenate(reco_list, dim=0)

            # Convert recommendations into context + actions
            reco_df = self._reco_tensor_to_df(reco_tensor, context_df)

        context_actions_df = self._reco_to_context_actions(reco_df, context_df)
        return context_actions_df

    def predict_metrics(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes ELUC and change for each sample in a context_actions_df.
        """
        eluc_df = self.predictor.predict(context_actions_df)
        change_df = self.compute_percent_changed(context_actions_df)

        return eluc_df, change_df

    def prescribe_land_use(self, context_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Wrapper for prescribe method that loads a candidate from disk using an id.
        Valid kwargs:
            cand_id: str, the ID of the candidate to load
            results_dir: Path, the directory where the candidate is stored
        Then takes in a context dataframe and prescribes actions.
        """
        candidate = Candidate(**self.candidate_params)
        gen = int(kwargs["cand_id"].split("_")[0])
        state_dict = torch.load(kwargs["results_dir"] / f"{gen + 1}" / f"{kwargs['cand_id']}.pt")
        candidate.load_state_dict(state_dict)

        context_actions_df = self.prescribe(candidate, context_df)
        return context_actions_df
    