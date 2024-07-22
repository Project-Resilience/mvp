"""
Validation of input and output dataframes for predictor evaluation.
"""
import pandas as pd

class Validator():
    """
    Validates input and output dataframes for predictor evaluation.
    Context, actions, outcomes do not necessarily have to match the project's CAO_MAPPING. For example, if we are
    just evaluating ELUC we can just pass the single column as outcomes.
    """
    def __init__(self, context: list[str], actions: list[str], outcomes: list[str]):
        self.context = context
        self.actions = actions
        self.outcomes = outcomes

    def validate_input(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Verifies all the context and actions columns are in context_actions_df.
        Then removes outcomes from context_actions_df and returns a deep copy of it.
        """
        if not set(self.context + self.actions) <= set(context_actions_df.columns):
            not_seen = set(self.context + self.actions) - set(context_actions_df.columns)
            raise ValueError(f"Columns {not_seen} not found in input dataframe.")

        seen_outcomes = [col for col in self.outcomes if col in context_actions_df.columns]
        return context_actions_df.drop(columns=seen_outcomes).copy()

    def validate_output(self, context_actions_df: pd.DataFrame, outcomes_df: pd.DataFrame):
        """
        Makes sure the index of context_actions_df and outcomes_df match so we can compute metrics like MAE.
        Also checks if all outcomes are present in the outcomes_df.
        """
        if not context_actions_df.index.equals(outcomes_df.index):
            raise ValueError("Index of context_actions_df and outcomes_df do not match.")

        if not set(self.outcomes) == set(outcomes_df.columns):
            print(self.outcomes, outcomes_df.columns)
            not_seen = set(self.outcomes) - set(outcomes_df.columns)
            raise ValueError(f"Outcomes {not_seen} not found in output dataframe.")

        return True
