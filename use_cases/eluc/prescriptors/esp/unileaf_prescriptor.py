from typing import Any
from typing import Dict
from typing import List

from pathlib import Path

import pandas as pd
import numpy as np
from keras.models import load_model

from esp_sdk.esp_evaluator import EspEvaluator

from data import constants
from data.eluc_data import ELUCEncoder
from prescriptors.prescriptor import Prescriptor

class UnileafPrescriptor(EspEvaluator, Prescriptor):
    """
    An Unileaf Prescriptor makes prescriptions given an ESP candidate and a context DataFrame.
    It is also an EspEvaluator implementation that returns metrics for ESP candidates.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 evaluation_df: pd.DataFrame,
                 data_encoder: ELUCEncoder,
                 predictors):
        """
        Constructs a prescriptor evaluator
        :param config: the ESP experiment config dictionary
        :param evaluation_df: the encoded Pandas DataFrame to use to evaluate the candidates
        :param data_encoder: the DataEncoder used to encode the dataset
        :param predictors: the predictors this prescriptor relies on
        """
        # Instantiate EspEvaluator
        # Note: sets self.config
        super().__init__(config)

        # CAO
        self.cao_mapping = {"context": self.get_context_field_names(config),
                            "actions": self.get_action_field_names(config),
                            "outcomes": self.get_fitness_metrics(config)}
        self.context_df = evaluation_df[self.cao_mapping["context"]]
        self.row_index = self.context_df.index

        # Convert the context DataFrame to a format a NN can ingest
        self.context_as_nn_input = self.convert_to_nn_input(self.context_df)

        # Data encoder
        self.data_encoder = data_encoder

        # Predictors
        self.predictors = predictors

    @staticmethod
    def convert_to_nn_input(context_df: pd.DataFrame) -> List[np.ndarray]:
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

    def _reco_to_context_actions(self, reco_df: pd.DataFrame, encoded_context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a dataframe containing recommended land use proportions to a dataframe containing
        the context and prescribed actions.
        """
        # This is gacky but has to happen sooner or later
        reco_df = reco_df.reset_index(drop=True)
        encoded_context_df = encoded_context_df.reset_index(drop=True)

        context_df = self.data_encoder.decode_as_df(encoded_context_df)

        # Linear scaling is implemented here:
        # Do ReLU here since we no longer softmax
        reco_df = reco_df.clip(0, None)
        # If all outputs 0, set to be uniform
        reco_df[reco_df.sum(axis=1) == 0] = 1 # Could be any positive constant, 1 for simplicity
        prescribed_total_df = reco_df.sum(axis=1)
        prescribed_total_df = prescribed_total_df.replace(0, 1)
        # Since we are no longer using softmax, do a linear scaling
        reco_df = reco_df.div(prescribed_total_df, axis=0)

        # Scale encoded_reco_df to context_df minus primf/primn
        # Multiply proportions by sum of non primn/primf cols
        reco_df = reco_df.mul(context_df[constants.RECO_COLS].sum(axis=1), axis=0)

        # Compute the diff
        # Note: the index need to match in order to subtract. Otherwise we get NaN
        prescribed_actions_df = reco_df[constants.RECO_COLS].reset_index(drop=True) - context_df[constants.RECO_COLS].reset_index(drop=True)

        # Rename the columns to match what the predictor expects
        prescribed_actions_df = prescribed_actions_df.rename(constants.RECO_MAP, axis=1)
        prescribed_actions_df[constants.NO_CHANGE_COLS] = 0
        
        # Aggregate the context and actions dataframes.
        context_actions_df = pd.concat([context_df,
                                                prescribed_actions_df[constants.DIFF_LAND_USE_COLS]],
                                                axis=1)
        
        return context_actions_df

    def evaluate_candidate(self, candidate):
        """
        Evaluates a single Prescriptor candidate and returns its metrics.
        Implements the EspEvaluator interface
        :param candidate: a Keras neural network or rule based Prescriptor candidate
        :return metrics: A dictionary of {'metric_name': metric_value}
        """
        # Save candidate to local file for easy debug
        # candidate.save('prescriptor.h5')
        
        # Prescribe actions
        # Single action, recommended percentage for each land use type
        # Note: prescribed action is a softmax, NOT encoded in the same scale as the context
        prescribed_actions_df = self.prescribe(candidate)
        
        # Convert the softmax into a DataFrame
        reco_land_use_df = pd.DataFrame(prescribed_actions_df["reco_land_use"].tolist(),
                                columns=constants.RECO_COLS)
        
        context_actions_df = self._reco_to_context_actions(reco_land_use_df, self.context_df)

        # Compute the metrics
        metrics = self._compute_metrics(context_actions_df)
        return metrics
    
    def _compute_metrics(self, context_actions_df):
        """
        Computes metrics from the passed context/actions DataFrame using the instance's trained predictors.
        :param encoded_context_actions_df: a DataFrame of context / prescribed actions
        :return: A dictionary of {'metric_name': metric_value}
        """
        metrics = {}
        
        # Get the predicted ELUC from the predictors
        preds = self.predict_eluc(context_actions_df)
        metrics['ELUC'] = preds['ELUC'].mean()
        
        # Compute the % of change
        change_df = self.compute_percent_changed(context_actions_df)
        metrics['change'] = change_df['change'].mean()
        
        return metrics
    
    def predict_eluc(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts ELUC using the given predictor
        """
        predictor = self.predictors[0]
        preds = predictor.predict(context_actions_df)
        preds = preds.astype("float64")
        return preds
    
    def compute_percent_changed(self, context_actions_df):
        """
        Calculates what percent of usable land was changed from the context to the actions.
        """
        context_actions_df = context_actions_df.reset_index(drop=True)

        # Sum the positive diffs
        percent_changed = context_actions_df[context_actions_df[constants.DIFF_LAND_USE_COLS] > 0][constants.DIFF_LAND_USE_COLS].sum(axis=1)
        # Land usage is only a portion of that cell, e.g 0.8. Scale back to 1
        # So that percent changed really represent the percentage of change within the land use
        # portion of the cell
        # I.e. how much of the pie chart has changed?
        total_land_use = context_actions_df[constants.LAND_USE_COLS].sum(axis=1)
        total_land_use = total_land_use.replace(0, 1)
        percent_changed = percent_changed / total_land_use
        df = pd.DataFrame(percent_changed, columns=['change'])
        return df

    def prescribe(self, candidate, context_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generates prescriptions using the passed candidate and context
        :param candidate: an ESP candidate, either neural network or rules
        :param context_df: a DataFrame containing the context to prescribe for,
         or None to use the instance one
        :return: a DataFrame containing actions prescribed for each context
        """
        if context_df is None:
            # No context is provided, use the instance's one
            context_as_nn_input = self.context_as_nn_input
            row_index = self.row_index
        else:
            # Convert the context DataFrame to something more suitable for neural networks
            context_as_nn_input = self.convert_to_nn_input(context_df)
            # Use the context's row index
            row_index = context_df.index

        # Temporarily removed, may come back if we do rule-based prescription
        # is_rule_based = isinstance(candidate, RuleSet)
        # if is_rule_based:
        #     actions = self._prescribe_from_rules(candidate, context_as_nn_input)
        # else:
        #     actions = self._prescribe_from_nn(candidate, context_as_nn_input)
        actions = self._prescribe_from_nn(candidate, context_as_nn_input)

        # Convert the prescribed actions to a DataFrame
        prescribed_actions_df = pd.DataFrame(actions,
                                             columns=self.cao_mapping["actions"],
                                             index=row_index)
        return prescribed_actions_df

    def _prescribe_from_nn(self, candidate, context_as_nn_input: List[np.ndarray]) -> Dict[str, Any]:
        """
        Generates prescriptions using the passed neural network candidate and context
        :param candidate: a Keras neural network candidate
        :param context_as_nn_input: a numpy array containing the context to prescribe for
        :return: a dictionary of action name to action value or list of action values
        """
        # Get the prescribed actions
        prescribed_actions = candidate.predict(context_as_nn_input)
        actions = {}

        if self._is_single_action_prescriptor():
            # Put the single action in an array to process it like multiple actions
            prescribed_actions = [prescribed_actions]

        for i, action_col in enumerate(self.cao_mapping["actions"]):
            if self._is_scalar(prescribed_actions[i]):
                # We have a single row and this action is numerical. Convert it to a scalar.
                actions[action_col] = prescribed_actions[i].item()
            else:
                actions[action_col] = prescribed_actions[i].tolist()
        return actions

    def _is_single_action_prescriptor(self):
        """
        Checks how many Actions have been defined in the Context, Actions, Outcomes mapping.
        :return: True if only 1 action is defined, False otherwise
        """
        return len(self.cao_mapping["actions"]) == 1

    @staticmethod
    def _is_scalar(prescribed_action):
        """
        Checks if the prescribed action contains a single value, i.e. a scalar, or an array.
        A prescribed action contains a single value if it has been prescribed for a single context sample
        :param prescribed_action: a scalar or an array
        :return: True if the prescribed action contains a scalar, False otherwise.
        """
        return prescribed_action.shape[0] == 1 and prescribed_action.shape[1] == 1

    @staticmethod
    def get_context_field_names(config: Dict[str, Any]) -> List[str]:
        """
        Returns the list of Context column names
        :param config: the ESP experiment config dictionary
        :return: the list of Context column names
        """
        nn_inputs = config["network"]["inputs"]
        contexts = [nn_input["name"] for nn_input in nn_inputs]
        return contexts

    @staticmethod
    def get_action_field_names(config: Dict[str, Any]) -> List[str]:
        """
        Returns the list of Action column names
        :param config: the ESP experiment config dictionary
        :return: the list of Action column names
        """
        nn_outputs = config["network"]["outputs"]
        actions = [nn_output["name"] for nn_output in nn_outputs]
        return actions

    @staticmethod
    def get_fitness_metrics(config: Dict[str, Any]) -> List[str]:
        """
        Returns the list of fitness metric names (Outcomes) to optimize.
        :param config: the ESP experiment config dictionary
        :return: the list of fitness metric names
        """
        metrics = config["evolution"]["fitness"]
        fitness_metrics = [metric["metric_name"] for metric in metrics]
        return fitness_metrics
    
    def prescribe_land_use(self, cand_id: str, results_dir: Path, context_df: pd.DataFrame) -> pd.DataFrame:
        gen = int(cand_id.split('_')[0])
        candidate_filename = results_dir / f"{gen}" / f"{cand_id}.h5"
        candidate = load_model(candidate_filename, compile=False)
        
        encoded_context_df = self.data_encoder.encode_as_df(context_df)

        reco_land_use = self.prescribe(candidate, encoded_context_df)
        reco_df = pd.DataFrame(reco_land_use["reco_land_use"].tolist(), columns=constants.RECO_COLS)
        context_actions_df = self._reco_to_context_actions(reco_df, encoded_context_df)

        context_actions_df = context_actions_df.set_index(context_df.index)

        return context_actions_df
    
    def predict_metrics(self, context_actions_df: pd.DataFrame) -> tuple:
        eluc_df = self.predict_eluc(context_actions_df)
        change_df = self.compute_percent_changed(context_actions_df)

        return eluc_df, change_df
        

