"""
Basic script to train prescriptors using ESP.
Note: This is not open-source and requires the ESP-SDK.
This is left here because the original paper used this implementation.
"""
import argparse
import os
import json
from pathlib import Path

from esp_sdk.esp_service import EspService

from data.eluc_data import ELUCData
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from prescriptors.esp.unileaf_prescriptor import UnileafPrescriptor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=str, help="Experiment ID to use for training.", required=True)
    parser.add_argument("--version", type=str, help="Version to use for training.", required=True)
    parser.add_argument("--config_path", type=str, help="Path to prescriptor configuration.",
                        default="prescriptors/esp/unileaf_configs/config-loctime-crop-nosoft.json")
    parser.add_argument("--nn_path", type=str, help="Path to neural net predictor to load.",
                        default="predictors/neural_network/trained_models/no_overlap_nn")
    parser.add_argument("--n_samples", type=float, default=0.001,
                        help="How much of the dataset to use for training. \
                            If <1 uses a proportion of the dataset, \
                            otherwise uses a flat number.")
    args = parser.parse_args()

    print("Loading data...")
    dataset = ELUCData()

    print("Initializing predictor...")
    nnp = NeuralNetPredictor()
    print("Loading predictor...")
    nn_path = Path(args.nn_path)
    nnp.load(nn_path)

    # Set up ESP service
    esp_username = os.getenv('ESP_SERVICE_USER')
    esp_password = os.getenv('ESP_SERVICE_PASSWORD')
    if not esp_username or not esp_password:
        raise ValueError('ESP Service username and password not found.')
    else:
        print('ESP Service username and password found.')

    print("Running prescriptor training...")
    config_path = Path(args.config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        presc_config = json.load(f)
        presc_config["LEAF"]["experiment_id"] = args.experiment_id
        presc_config["LEAF"]["version"] = args.version

    eval_df_encoded = dataset.get_encoded_train()
    if args.n_samples:
        if args.n_samples < 1:
            eval_df_encoded = eval_df_encoded.sample(frac=args.n_samples, random_state=42)
        else:
            eval_df_encoded = eval_df_encoded.sample(n=int(args.n_samples), random_state=42)
    esp_service = EspService(presc_config, esp_username, esp_password)
    esp_evaluator = UnileafPrescriptor(presc_config,
                                        eval_df_encoded,
                                        dataset.encoder,
                                        [nnp])
    experiment_results_dir = esp_service.train(esp_evaluator)