"""
Serializer for LandUsePrescriptor.
"""
import json
from pathlib import Path

from prsdk.persistence.serializers.serializer import Serializer
import torch

from data.eluc_encoder import ELUCEncoder
from prescriptors.nsga2.candidate import Candidate
from prescriptors.nsga2.land_use_prescriptor import LandUsePrescriptor


class PrescriptorSerializer(Serializer):
    """
    Serializer in charge of saving single prescriptor model from LandUsePrescriptor.
    Keeps track of neural network candidate params, encoder fields, and model state dict.
    """
    def save(self, model: LandUsePrescriptor, path: Path):
        """
        Saves the prescriptor to disk.
        """
        path.mkdir(parents=True, exist_ok=True)
        cand_params = {
            "in_size": model.candidate.in_size,
            "hidden_size": model.candidate.hidden_size,
            "out_size": model.candidate.out_size
        }
        with open(path / "cand_params.json", "w", encoding="utf-8") as file:
            json.dump(cand_params, file)
        model.encoder.save_fields(path / "fields.json")
        torch.save(model.candidate.state_dict(), path / "model.pt")

    def load(self, path: Path) -> LandUsePrescriptor:
        """
        Loads a prescriptor from disk.
        """
        with open(path / "cand_params.json", "r", encoding="utf-8") as file:
            cand_params = json.load(file)
        candidate = Candidate(**cand_params)
        candidate.load_state_dict(torch.load(path / "model.pt"))
        encoder = ELUCEncoder.from_json(path / "fields.json")
        model = LandUsePrescriptor(candidate, encoder)
        return model
