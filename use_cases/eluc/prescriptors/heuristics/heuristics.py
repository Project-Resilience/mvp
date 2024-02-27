"""
Heuristic to compare our prescriptors to.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from data import constants
from predictors.predictor import Predictor
from prescriptors.prescriptor import Prescriptor

class HeuristicPrescriptor(Prescriptor, ABC):
    def __init__(self, pct: float, predictor: Predictor):
        self.pct = pct
        self.predictor = predictor

    @abstractmethod
    def _reco_heuristic(self, context_df):
        raise NotImplementedError
    
    def prescribe_land_use(self, _, __, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        The header of this function is kinda gacky. Not sure what to do about this...
        """
        reco_df = self._reco_heuristic(context_df)
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
        change_df = self._compute_percent_changed(context_actions_df)
        return eluc_df, change_df


class EvenHeuristic(HeuristicPrescriptor):

    def __init__(self, pct: float, best_col: str, predictor: Predictor):
        super().__init__(pct, predictor)
        self.best_col = best_col
        self.presc_cols = [col for col in constants.RECO_COLS if col != best_col]

    def presc_fn(self, col, row_sum, max_change):
        return col - (col/row_sum) * max_change if col - (col/row_sum) * max_change >= 0 else 0

    def _even_prescribe_row(self, row):
        pass
    
    def _reco_heuristic(self, context_df):
        """
        Takes evenly from all columns and adds to best col
        """
        adjusted = context_df.copy()
        adjusted["scaled_change"] = self.pct * adjusted[constants.LAND_USE_COLS].sum(axis=1)
        adjusted["row_sum"] = adjusted[self.presc_cols].sum(axis=1)
        adjusted["max_change"] = adjusted[["scaled_change", "row_sum"]].min(axis=1)
        # Reduce all columns by even amount
        for col in self.presc_cols:
            adjusted[col] -= adjusted["max_change"] / len(self.presc_cols)
        # Increase best column by max change
        adjusted[self.best_col] = adjusted[self.best_col] + adjusted["max_change"]
        return adjusted.drop(["scaled_change", "row_sum", "max_change"], axis=1)

class PerfectHeuristic(HeuristicPrescriptor):

    def __init__(self, pct: float, coefs: list[float], predictor: Predictor):
        super().__init__(pct, predictor)
        coefficients = zip(constants.LAND_USE_COLS, coefs)
        self.reco_coefs = [coef for coef in coefficients if coef[0] in constants.RECO_COLS]
        self.reco_coefs = sorted(self.reco_coefs, key=lambda x: x[1], reverse=True)

    def _perfect_prescribe_row(self, row):
        # Weird case where row is all zeroes
        if row[constants.LAND_USE_COLS].sum() == 0:
            return row
        scaled_change = self.pct * row[constants.LAND_USE_COLS].sum()
        best_col = self.reco_coefs[-1][0]
        max_change = min(row[constants.RECO_COLS].sum() - row[best_col], scaled_change)
        changed = 0
        for coef in self.reco_coefs:
            if not coef[0] == best_col: # This technically shouldn't be necessary
                # If we have more change left than there is in this column, delete it all
                if row[coef[0]] < max_change - changed:
                    changed += row[coef[0]]
                    row[coef[0]] = 0
                # Otherwise, remove however much change is left
                else:
                    row[coef[0]] -= (max_change - changed)
                    changed = max_change
                    break
        row[best_col] += changed
        return row

    def _reco_heuristic(self, context_df: pd.DataFrame):
        """
        Perfect prescription algorithm:
            1. Add to forest as much pct change as possible up to forest = 1
            2. Subtract from bottom up however much was changed
        """

        adjusted = context_df.copy()
        adjusted = adjusted.apply(self._perfect_prescribe_row, axis=1)
        return adjusted
    
