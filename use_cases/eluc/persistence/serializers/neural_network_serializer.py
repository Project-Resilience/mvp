"""
Serializer for the Neural Network Predictor class.
"""
import json
from pathlib import Path

import joblib
from prsdk.persistence.serializers.serializer import Serializer
import torch

from predictors.neural_network.eluc_neural_net import ELUCNeuralNet
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor


class NeuralNetSerializer(Serializer):
    """
    Serializer for the NeuralNetPredictor.
    Saves config necessary to recreate the model, the model itself, and the scaler for the data to a folder.
    """
    def save(self, model: NeuralNetPredictor, path: Path):
        """
        Saves model, config, and scaler into format for loading.
        Generates path to folder if it does not exist.
        :param path: path to folder to save model files.
        """
        if model.model is None:
            raise ValueError("Model not fitted yet.")
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "features": model.features,
            "label": model.label,
            "hidden_sizes": model.hidden_sizes,
            "linear_skip": model.linear_skip,
            "dropout": model.dropout,
            "device": model.device,
            "epochs": model.epochs,
            "batch_size": model.batch_size,
            "optim_params": model.optim_params,
            "train_pct": model.train_pct,
            "step_lr_params": model.step_lr_params
        }
        with open(path / "config.json", "w", encoding="utf-8") as file:
            json.dump(config, file)
        torch.save(model.model.state_dict(), path / "model.pt")
        joblib.dump(model.scaler, path / "scaler.joblib")

    def load(self, path: Path) -> "NeuralNetPredictor":
        """
        Loads a model from a given folder. Creates empty model with config, then loads model state dict and scaler.
        :param path: path to folder containing model files.
        """
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Path {path} does not exist.")
        if not (path / "config.json").exists() or \
            not (path / "model.pt").exists() or \
                not (path / "scaler.joblib").exists():
            raise FileNotFoundError("Model files not found in path.")

        # Initialize model with config
        with open(path / "config.json", "r", encoding="utf-8") as file:
            config = json.load(file)

        nnp = NeuralNetPredictor(config)

        nnp.model = ELUCNeuralNet(len(config["features"]),
                                  config["hidden_sizes"],
                                  config["linear_skip"],
                                  config["dropout"])
        nnp.model.load_state_dict(torch.load(path / "model.pt"))
        nnp.model.to(config["device"])
        nnp.model.eval()
        nnp.scaler = joblib.load(path / "scaler.joblib")
        return nnp
