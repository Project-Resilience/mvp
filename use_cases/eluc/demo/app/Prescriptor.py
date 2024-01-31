import os
import json
from typing import List
import pandas as pd
import numpy as np
from keras.models import load_model

from . import constants
from . import utils


class Prescriptor:
    """
    Wrapper for Keras prescriptor and encoder.
    """

    def __init__(self, prescriptor_id: str):
        """
        :param prescriptor_id: ID of Keras prescriptor to load.
        """
        prescriptor_model_filename = os.path.join(constants.PRESCRIPTOR_PATH,
                                                prescriptor_id + '.h5')

        self.prescriptor_model = load_model(prescriptor_model_filename, compile=False)

        self.encoder = None
        with open(constants.FIELDS_PATH, 'r') as f:
            fields = json.load(f)
            self.encoder = utils.Encoder(fields)


    def _is_single_action_prescriptor(self, actions):
        """
        Checks how many Actions have been defined in the Context, Actions, Outcomes mapping.
        :return: True if only 1 action is defined, False otherwise
        """
        return len(actions) == 1

    def _is_scalar(self, prescribed_action):
        """
        Checks if the prescribed action contains a single value, i.e. a scalar, or an array.
        A prescribed action contains a single value if it has been prescribed for a single context sample
        :param prescribed_action: a scalar or an array
        :return: True if the prescribed action contains a scalar, False otherwise.
        """
        return prescribed_action.shape[0] == 1 and prescribed_action.shape[1] == 1

    def _convert_to_nn_input(self, context_df: pd.DataFrame) -> List[np.ndarray]:
        """
        Converts a context DataFrame to a list of numpy arrays a neural network can ingest
        :param context_df: a DataFrame containing inputs for a neural network. Number of inputs and size must match
        :return: a list of numpy ndarray, on ndarray per neural network input
        """
        # The NN expects a list of i inputs by s samples (e.g. 9 x 299).
        # So convert the data frame to a numpy array (gives shape 299 x 9), transpose it (gives 9 x 299)
        # and convert to list(list of 9 arrays of 299)
        context_as_nn_input = list(context_df.to_numpy().transpose())
        # Convert each column's list of 1D array to a 2D array
        context_as_nn_input = [np.stack(context_as_nn_input[i], axis=0) for i in
                            range(len(context_as_nn_input))]
        return context_as_nn_input

    def __prescribe_from_model(self, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates prescriptions using the passed neural network candidate and context
        ::param context_df: a DataFrame containing the context to prescribe for,
        :return: a pandas DataFrame of action name to action value or list of action values
        """
        action_list = ['reco_land_use']

        # Convert the input df
        context_as_nn_input = self._convert_to_nn_input(context_df)
        row_index = context_df.index

        # Get the prescrib?ed actions
        prescribed_actions = self.prescriptor_model.predict(context_as_nn_input)
        actions = {}

        if self._is_single_action_prescriptor(action_list):
            # Put the single action in an array to process it like multiple actions
            prescribed_actions = [prescribed_actions]

        for idx, action_col in enumerate(action_list):
            if self._is_scalar(prescribed_actions[idx]):
                # We have a single row and this action is numerical. Convert it to a scalar.
                actions[action_col] = prescribed_actions[idx].item()
            else:
                actions[action_col] = prescribed_actions[idx].tolist()

        # Convert the prescribed actions to a DataFrame
        prescribed_actions_df = pd.DataFrame(actions,
                                            columns=action_list,
                                            index=row_index)
        return prescribed_actions_df


    def run_prescriptor(self, sample_context_df):
        """
        Runs prescriptor on context. Then re-scales prescribed land
        use to match how much was used in the sample.

        :param sample_context_df: a DataFrame containing the context
        :return: DataFrame of prescribed land use
        """
        encoded_sample_context_df = self.encoder.encode_as_df(sample_context_df)
        prescribed_actions_df = self.__prescribe_from_model(encoded_sample_context_df)
        reco_land_use_df = pd.DataFrame(prescribed_actions_df["reco_land_use"].tolist(),
                                    columns=constants.RECO_COLS)

        # We removed softmax from model so we have to scale them to sum to 1
        reco_land_use_df = reco_land_use_df.clip(0, None)
        reco_land_use_df[reco_land_use_df.sum(axis=1) == 0] = 1
        reco_land_use_df = reco_land_use_df.div(reco_land_use_df.sum(axis=1), axis=0)

        # Re-scales our prescribed land to match the amount of land used in the sample
        used = sample_context_df[constants.RECO_COLS].iloc[0].sum()
        reco_land_use_df = reco_land_use_df[constants.RECO_COLS].mul(used, axis=0)

        # Reorder columns
        return reco_land_use_df[constants.RECO_COLS]
