"""
Script used to train NSGA-II prescriptors.
Requires a config file with the same fields as shown in the
test.json file in prescriptors/nsga2/configs
"""
import argparse
import json
from pathlib import Path

from prsdk.persistence.serializers.neural_network_serializer import NeuralNetSerializer

from data.eluc_data import ELUCData
from data.eluc_encoder import ELUCEncoder
from prescriptors.nsga2.trainer import TorchTrainer
from predictors.percent_change.percent_change_predictor import PercentChangePredictor
from predictors.percent_change.crop_change_predictor import CropChangePredictor


if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    with open(Path(args.config_path), "r", encoding="utf-8") as f:
        config = json.load(f)

    print("Loading dataset...")
    dataset = ELUCData.from_hf()
    encoder = ELUCEncoder.from_pandas(dataset.train_df)

    print("Loading predictors...")
    # TODO: We need to make it so you can load any predictor here
    nnp_serializer = NeuralNetSerializer()
    nnp = nnp_serializer.load(Path(config["predictor_path"]))
    pct_change = PercentChangePredictor()
    crop_change = CropChangePredictor()
    predictors = {"ELUC": nnp, "change": pct_change, "cropchange": crop_change}

    print("Initializing prescription...")
    if "seed_dir" in config["evolution_params"].keys():
        config["evolution_params"]["seed_dir"] = Path(config["evolution_params"]["seed_dir"])
    tp = TorchTrainer(
        eval_df=dataset.train_df.sample(frac=0.001, random_state=42),
        encoder=encoder,
        predictors=predictors,
        batch_size=4096,
        **config["evolution_params"]
    )
    print("Training prescriptors...")
    save_path = Path(config["save_path"])
    final_pop = tp.neuroevolution(save_path)
    print("Done!")
