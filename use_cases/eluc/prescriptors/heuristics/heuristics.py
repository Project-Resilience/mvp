"""
Heuristic to compare our prescriptors to.
"""
from abc import ABC, abstractmethod

import pandas as pd

from data import constants
from predictors.predictor import Predictor
from prescriptors.prescriptor import Prescriptor

class HeuristicPrescriptor(Prescriptor, ABC):
    """
    Abstract heuristic prescriptor class that inherits from prescriptor class.
    Has a percentage threshold that the heuristic is to reach but not exceed.
    Also takes a predictor so that we can evaluate metrics.
    Requires an implementation of reco_heuristic which takes a context dataframe and returns
    recommendations based on the heuristic.
    """
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

    @abstractmethod
    def _reco_heuristic(self, pct: float, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method that takes a percentage threshold of land change and a 
        context dataframe and returns a dataframe of recommendations based on the heuristic.
        """
        raise NotImplementedError

    def prescribe_land_use(self, context_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Implementation of prescribe_land_use using a heuristic. Calls the implementation of _reco_heuristic.
        Kwargs must contain a "pct" key that is the percentage of land-use change to prescribe up to.
        """
        reco_df = self._reco_heuristic(kwargs["pct"], context_df)
        prescribed_actions_df = reco_df[constants.RECO_COLS] - context_df[constants.RECO_COLS]

        # Rename the columns to match what the predictor expects
        prescribed_actions_df = prescribed_actions_df.rename(constants.RECO_MAP, axis=1)
        prescribed_actions_df[constants.NO_CHANGE_COLS] = 0

        # Aggregate the context and actions dataframes.
        context_actions_df = pd.concat([context_df, prescribed_actions_df[constants.DIFF_LAND_USE_COLS]], axis=1)
        return context_actions_df

    def predict_metrics(self, context_actions_df: pd.DataFrame) -> tuple:
        column_order = constants.CAO_MAPPING["context"] + constants.CAO_MAPPING["actions"]
        eluc_df = self.predictor.predict(context_actions_df[column_order])
        change_df = self.compute_percent_changed(context_actions_df)
        return eluc_df, change_df


class EvenHeuristic(HeuristicPrescriptor):
    """
    Implementation of HeuristicPrescriptor that evenly distributes land use to a "best" column.
    """
    def __init__(self, best_col: str, predictor: Predictor):
        super().__init__(predictor)
        self.best_col = best_col
        self.presc_cols = [col for col in constants.RECO_COLS if col != best_col]

    def _reco_heuristic(self, pct: float, context_df: pd.DataFrame):
        """
        Takes evenly from all columns and adds to best col.
        Removes land_use * (total_change / changeable_land) so we remove proportionally
        rather than truly evenly.
        """
        adjusted = context_df.copy()
        adjusted["scaled_change"] = pct * adjusted[constants.LAND_USE_COLS].sum(axis=1)
        adjusted["row_sum"] = adjusted[self.presc_cols].sum(axis=1)
        to_change = adjusted["row_sum"] > 0
        adjusted["max_change"] = adjusted[["scaled_change", "row_sum"]].min(axis=1)

        max_change = adjusted.loc[to_change, "max_change"]
        row_sum = adjusted.loc[to_change, "row_sum"]
        # Reduce all columns by even amount
        for col in self.presc_cols:
            adjusted.loc[to_change, col] -= adjusted.loc[to_change, col] * max_change / row_sum
        # Increase best column by max change
        adjusted.loc[to_change, self.best_col] = adjusted.loc[to_change, self.best_col] + max_change
        adjusted = adjusted.drop(["scaled_change", "row_sum", "max_change"], axis=1)
        return adjusted

class PerfectHeuristic(HeuristicPrescriptor):
    """
    Implementation of HeuristicPrescriptor that does an informed land use prescription 
    based on linear regression coefficients.
    """
    def __init__(self, coefs: list[float], predictor: Predictor):
        """
        We save and sort the columns by highest coefficient i.e. most emissions.
        Separate the best column according to the coefficients to add to.
        """
        super().__init__(predictor)
        assert len(coefs) == len(constants.RECO_COLS)
        # Sort columns by coefficient
        reco_cols = list(constants.RECO_COLS)
        zipped = zip(reco_cols, coefs)
        sorted_zip = sorted(zipped, key=lambda x: x[1], reverse=True)
        self.reco_cols, _ = zip(*sorted_zip)
        self.reco_cols = list(self.reco_cols)

        # Take best column and remove from reco cols
        self.best_col = self.reco_cols[-1]
        self.reco_cols.pop()

    def _reco_heuristic(self, pct: float, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perfect prescription algorithm:
            1. Subtract up to scaled change starting from worst column.
            2. Add to best column however much was removed.
        It is done in a vectorized way to massively speed up computation.
        """
        adjusted = context_df.copy()

        adjusted["scaled_change"] = pct * adjusted[constants.LAND_USE_COLS].sum(axis=1)
        adjusted["presc_sum"] = adjusted[self.reco_cols].sum(axis=1)
        adjusted["amt_change"] = adjusted[["scaled_change", "presc_sum"]].min(axis=1)

        for col in self.reco_cols:
            change = adjusted[[col, "amt_change"]].min(axis=1)
            adjusted[col] -= change
            adjusted["amt_change"] -= change

        adjusted[self.best_col] += adjusted[["scaled_change", "presc_sum"]].min(axis=1)
        adjusted = adjusted.drop(["scaled_change", "presc_sum", "amt_change"], axis=1)
        return adjusted
