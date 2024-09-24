"""
Utility functions for the demo application.
"""
import json
from pathlib import Path

import pandas as pd
import torch

from prsdk.persistence.serializers.neural_network_serializer import NeuralNetSerializer
from prsdk.persistence.persistors.hf_persistor import HuggingFacePersistor

import app.constants as app_constants
from data import constants
from data.eluc_encoder import ELUCEncoder
from prescriptors.prescriptor_manager import PrescriptorManager
from prescriptors.nsga2.candidate import Candidate
from prescriptors.nsga2.land_use_prescriptor import LandUsePrescriptor
from predictors.percent_change.percent_change_predictor import PercentChangePredictor

def add_nonland(context_actions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a nonland column to the context_actions_df
    """
    updated_df = context_actions_df.copy()
    updated_df["nonland"] = 1 - updated_df[constants.LAND_USE_COLS].sum(axis=1)
    updated_df["nonland"] = updated_df["nonland"].clip(lower=0)
    return updated_df


class EvolutionHandler:
    """
    Class that handles running our trained prescriptors
    """

    def __init__(self):
        self.prescriptor_path = Path("app/results/prescriptors")
        self.prescriptor_list = [model_path.stem for model_path in self.prescriptor_path.glob("*.pt")]
        def sort_cand_id(cand_id: str):
            return (int(cand_id.split("_")[0]), int(cand_id.split("_")[1]))
        self.prescriptor_list = sorted(self.prescriptor_list, key=sort_cand_id)

        # Load prescriptors
        self.prescriptor_manager = self.load_prescriptors()

    def load_prescriptors(self) -> tuple[list[str], PrescriptorManager]:
        """
        Loads in prescriptors from disk, downloads from HuggingFace first if needed.
        TODO: Currently hard-coded to load specific prescriptors from pareto path.
        :return: dict of prescriptor name -> prescriptor object.
        """
        with open("app/results/fields.json", "r", encoding="utf-8") as f:
            fields = json.load(f)
        encoder = ELUCEncoder(fields)

        candidate_params = {"in_size": len(constants.CAO_MAPPING["context"]),
                            "hidden_size": 16,
                            "out_size": len(constants.RECO_COLS)}
        prescriptors = {}
        for cand_id in self.prescriptor_list:
            cand_path = self.prescriptor_path / f"{cand_id}.pt"
            candidate = Candidate(**candidate_params, cand_id=cand_id)
            candidate.load_state_dict(torch.load(cand_path))
            prescriptors[cand_id] = LandUsePrescriptor(candidate, encoder)

        pred_persistor = HuggingFacePersistor(NeuralNetSerializer())
        eluc_predictor = pred_persistor.from_pretrained("danyoung/eluc-global-nn", local_dir="app/results/predictor")
        predictors = {"ELUC": eluc_predictor, "change": PercentChangePredictor()}

        prescriptor_manager = PrescriptorManager(prescriptors, predictors)

        return prescriptor_manager
    
    

    def prescribe_all(self, context_df: pd.DataFrame):
        """
        Runs all prescriptors on the given context_df, returning prescriptions and outcomes.
        context_df should be a single row so we can concatenate the results into a single df.
        """
        assert len(context_df) == 1, f"Context should be a single row, got {len(context_df)}"
        context_actions_dfs = []
        outcomes_dfs = []
        for cand_id in self.prescriptor_list:
            context_actions_df = self.prescriptor_manager.prescribe(cand_id, context_df)
            outcomes_df = self.prescriptor_manager.predict_metrics(context_actions_df)
            context_actions_df["cand_id"] = cand_id
            context_actions_df = add_nonland(context_actions_df)
            outcomes_df["cand_id"] = cand_id
            context_actions_dfs.append(context_actions_df)
            outcomes_dfs.append(outcomes_df)

        # Concatenate the results together with each entry in list as a row
        total_context_actions_df = pd.concat(context_actions_dfs, axis=0)
        total_outcomes_df = pd.concat(outcomes_dfs, axis=0)

        results_df = pd.merge(total_context_actions_df, total_outcomes_df, how="left", on="cand_id")
        return results_df

    def context_actions_to_recos(self, context_actions_df: pd.DataFrame):
        """
        Reverts context actions back to recommendations
        """
        reco_diff_df = context_actions_df[constants.DIFF_RECO_COLS]
        reversed_map = {v: k for k, v in constants.RECO_MAP.items()}
        reco_diff_df = reco_diff_df.rename(reversed_map, axis=1)

        reco_df = context_actions_df[constants.RECO_COLS].copy() + reco_diff_df
        reco_df["cand_id"] = context_actions_df["cand_id"]
        
        return reco_df
