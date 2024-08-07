"""
Class defining the ELUC problem in Pymoo.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from prsdk.data.torch_data import TorchDataset
from prsdk.predictors.predictor import Predictor

from data import constants


class ELUCProblem(ElementwiseProblem):
    """
    Pymoo problem class for ELUC optimization.
    Takes in a dataframe of evaluation data, the neural network parameters, and a list of predictors to evaluate with.
    """
    def __init__(self,
                 eval_df: pd.DataFrame,
                 nn_params: dict[str, int],
                 predictors: list[Predictor],
                 batch_size: int,
                 device: str = "cpu"):
        num_params = (nn_params["in_size"]+1) * nn_params["hidden_size"] + \
            (nn_params["hidden_size"]+1) * nn_params["out_size"]
        super().__init__(n_var=num_params, n_obj=2, n_constr=0, xl=[-1] * num_params, xu=[1] * num_params)

        self.device = device
        self.predictors = predictors
        self.nn_params = nn_params
        self.eval_df = eval_df
        self.scaler = StandardScaler()
        encoded_context = self.scaler.fit_transform(eval_df[constants.CAO_MAPPING["context"]])
        encoded_context_ds = TorchDataset(encoded_context,
                                          np.zeros((len(encoded_context), len(constants.RECO_COLS))))
        self.eval_loader = DataLoader(encoded_context_ds, batch_size=batch_size, shuffle=False)

    def var_to_model(self, x):
        """
        Converts a flattened numpy array of parameters to a fixed architecture PyTorch model.
        """
        flattened = torch.Tensor(x)
        state_dict = OrderedDict()
        param_count = 0

        state_dict["0.weight"] = flattened[:self.nn_params["in_size"] * self.nn_params["hidden_size"]].reshape(
            self.nn_params["hidden_size"], self.nn_params["in_size"])
        param_count += self.nn_params["in_size"] * self.nn_params["hidden_size"]

        state_dict["0.bias"] = flattened[param_count:param_count + self.nn_params["hidden_size"]]
        param_count += self.nn_params["hidden_size"]

        state_dict["2.weight"] = flattened[param_count:param_count + self.nn_params["hidden_size"]
                                           * self.nn_params["out_size"]].reshape(
                                               self.nn_params["out_size"], self.nn_params["hidden_size"])
        param_count += self.nn_params["hidden_size"] * self.nn_params["out_size"]

        state_dict["2.bias"] = flattened[param_count:param_count + self.nn_params["out_size"]]
        
        model = torch.nn.Sequential(
            torch.nn.Linear(self.nn_params["in_size"], self.nn_params["hidden_size"]),
            torch.nn.Tanh(),
            torch.nn.Linear(self.nn_params["hidden_size"], self.nn_params["out_size"]),
            torch.nn.Softmax(dim=1))
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model
    
    def _reco_tensor_to_df(self, reco_tensor: torch.Tensor, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts raw Candidate neural network output tensor to scaled dataframe.
        Sets the indices of the recommendations so that we can subtract from the context to get
        the land diffs.
        """
        reco_df = pd.DataFrame(reco_tensor.cpu().numpy(), index=context_df.index, columns=constants.RECO_COLS)
        reco_df = reco_df.clip(0, None)  # ReLU
        reco_df[reco_df.sum(axis=1) == 0] = 1  # Rows of all 0s are set to 1s
        reco_df = reco_df.div(reco_df.sum(axis=1), axis=0)  # Normalize to sum to 1
        reco_df = reco_df.mul(context_df[constants.RECO_COLS].sum(axis=1), axis=0)  # Rescale to match original sum
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
                                        presc_actions_df[constants.CAO_MAPPING["actions"]]], axis=1)
        return context_actions_df

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluation function for Pymoo. Converts parameters to a model, then evaluates on the evaluation data.
        """
        candidate = self.var_to_model(x)
        outputs = []
        with torch.no_grad():
            for batch in self.eval_loader:
                batch = batch[0].to(self.device)
                outputs.append(candidate(batch))
            outputs = torch.cat(outputs, dim=0)
            reco_df = self._reco_tensor_to_df(outputs, self.eval_df)
            context_actions_df = self._reco_to_context_actions(reco_df, self.eval_df)

        out["F"] = [predictor.predict(context_actions_df).mean() for predictor in self.predictors]
