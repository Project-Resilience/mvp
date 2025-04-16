from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
import torch
from tqdm import tqdm

from experiments.utils.pareto import get_overall_pareto_df
from prsdk.predictors.predictor import Predictor

from data import constants
from data.eluc_encoder import ELUCEncoder
from prescriptors.nsga2.candidate import Candidate
from prescriptors.nsga2.land_use_prescriptor import LandUsePrescriptor
from prescriptors.prescriptor_manager import PrescriptorManager


def load_candidate(results_dir: Path, cand_id: str, cand_params: dict[str, int], old=False) -> Candidate:
    gen = int(cand_id.split('_')[0])
    if old:
        gen += 1
    cand_path = results_dir / str(gen) / f"{cand_id}.pt"
    cand = Candidate(**cand_params, device="mps", cand_id=cand_id)
    cand.load_state_dict(torch.load(cand_path, map_location="cpu"))
    return cand


def create_experiment_manager(results_dir: Path,
                              final_gen: int,
                              predictors: dict[str, Predictor],
                              encoder: ELUCEncoder,
                              outcomes: list[str],
                              old=False) -> PrescriptorManager:
    """
    Creates a prescriptor manager from the final Pareto front of a given experiment.
    """
    all_pareto_df = get_overall_pareto_df(final_gen, results_dir, outcomes)
    cand_ids = all_pareto_df["id"].tolist()

    # TODO: This is hard-coded for now
    candidate_params = {"in_size": len(constants.CAO_MAPPING["context"]),
                        "hidden_size": 16,
                        "out_size": len(constants.RECO_COLS)}

    cands = [load_candidate(results_dir, cand_id, candidate_params, old) for cand_id in cand_ids]
    prescs = {cand.cand_id: LandUsePrescriptor(cand, encoder) for cand in cands}
    return PrescriptorManager(prescs, predictors)


def evaluate_experiment(results_dir: Path,
                        final_gen: int,
                        context_df: pd.DataFrame,
                        encoder: ELUCEncoder,
                        outcomes: list[str],
                        predictors: dict[str, Predictor],
                        old=False) -> pd.DataFrame:
    torch_manager = create_experiment_manager(results_dir, final_gen, predictors, encoder, outcomes, old)
    cand_ids = list(torch_manager.prescriptors.keys())
    return evaluate_prescriptors(torch_manager, cand_ids, context_df, outcomes)


def evaluate_prescriptors(prescriptor_manager: PrescriptorManager,
                          cand_ids: list[str],
                          context_df: pd.DataFrame,
                          outcomes: list[str]) -> pd.DataFrame:
    rows = []
    for cand_id in tqdm(cand_ids):
        assert cand_id in prescriptor_manager.prescriptors
        context_actions_df = prescriptor_manager.prescribe(cand_id, context_df)
        outcome_df = prescriptor_manager.predict_metrics(context_actions_df)
        # return [outcome_df[outcome].mean() for outcome in outcomes]
        row = {outcome: outcome_df[outcome].mean() for outcome in outcomes}
        row["cand_id"] = cand_id
        rows.append(row)

    return pd.DataFrame(rows)


def trained_prescribe_and_predict(prescriptor_manager: PrescriptorManager, cand_id: str, context_df: pd.DataFrame):
    context_actions_df = prescriptor_manager.prescribe(cand_id, context_df)
    outcome_df = prescriptor_manager.predict_metrics(context_actions_df)
    context_actions_df["ELUC"] = outcome_df["ELUC"]
    context_actions_df["change"] = outcome_df["change"]
    # context_actions_df["cropchange"] = outcome_df["cropchange"]
    return context_actions_df


def closest_cand_id(results_df: pd.DataFrame, change: float):
    """
    Gets the cand_id of the row with the closest change to the given change that is greater than the given change.
    """
    more_change = results_df[results_df["change"] > change]
    # We have more change than all the candidates. Just pick the last one
    if len(more_change) == 0:
        return results_df.iloc[results_df["change"].idxmax()]["cand_id"]
    idx = more_change["change"].idxmin()
    if idx == -1:
        return -1
    return results_df.iloc[idx]["cand_id"]


def check_non_dominated(row: pd.Series, results_df: pd.DataFrame, outcomes: list[str]):
    """
    Checks if the row is dominated by any row in the results_df
    """
    for _, compare in results_df.iterrows():
        # See if compare dominates row
        dominates = False
        for outcome in outcomes:
            if compare[outcome] < row[outcome]:
                dominates = True
            if compare[outcome] > row[outcome]:
                dominates = False
                break

        # If compare dominates row, return False since we are not non-dominated
        if dominates:
            return False

    # If we make it all the way through without being dominated return true
    return True


def get_hypervolume(results_df: pd.DataFrame, ref_point: np.ndarray, ideal: np.ndarray, outcomes: list[str]):
    metric = Hypervolume(ref_point=ref_point,
                         norm_ref_point=True,
                         zero_to_one=True,
                         ideal=ideal,
                         nadir=ref_point)
    F = results_df[outcomes].values
    return metric.do(F)
