"""
Script used to train NSGA-II prescriptors.
Requires a config file with the same fields as shown in the
test.json file in prescriptors/nsga2/configs
"""

import argparse
import json
from pathlib import Path

from data.eluc_data import ELUCData
from prescriptors.nsga2.torch_prescriptor import TorchPrescriptor
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor

if __name__ == "__main__":

    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    with open(Path(args.config_path), "r", encoding="utf-8") as f:
        config = json.load(f)

    print("Loading dataset...")
    dataset = ELUCData()

    print("Loading predictor...")
    # TODO: We need to make it so you can load any predictor here
    nnp = NeuralNetPredictor()
    nnp_path = Path(config["predictor_path"])
    nnp.load(nnp_path)

    print("Initializing prescription...")
    if "seed_dir" in config.keys():
        config["seed_dir"] = Path(config["seed_dir"])
    tp = TorchPrescriptor(
        eval_df=dataset.train_df.sample(frac=0.001, random_state=42),
        encoder=dataset.encoder,
        predictor=nnp,
        batch_size=4096,
        **config["evolution_params"]
    )
    print("Training prescriptors...")
    save_path = Path(config["save_path"])
    final_pop = tp.neuroevolution(save_path)
    print("Done!")