"""
Script used to train NSGA-II prescriptors.
Requires a config file with the same fields as shown in the
test.json file in prescriptors/nsga2/configs
"""
import argparse
import json
from pathlib import Path

from prsdk.persistence.persistors.hf_persistor import HuggingFacePersistor
from prsdk.persistence.serializers.neural_network_serializer import NeuralNetSerializer

from data.min_eluc_data import MinimalELUCData
from data.eluc_encoder import ELUCEncoder
from prescriptors.nsga2.trainer import TorchTrainer
from predictors.percent_change.percent_change_predictor import PercentChangePredictor
from predictors.percent_change.crop_change_predictor import CropChangePredictor
from prescriptors.nsga2.create_seeds import seed_base, seed_rhea


if __name__ == "__main__":
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_paths", type=str, nargs="+", default=[], required=True)
    parser.add_argument("--n_repeats", type=int, default=1)
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = MinimalELUCData.from_hf()
    encoder = ELUCEncoder.from_pandas(dataset.train_df)

    print("Loading predictors...")
    # TODO: We need to make it so you can load any predictor here
    pred_persistor = HuggingFacePersistor(NeuralNetSerializer())
    nnp = pred_persistor.from_pretrained("danyoung/eluc-global-nn",
                                         local_dir="predictors/trained_models/danyoung--eluc-global-nn")
    pct_change = PercentChangePredictor()
    # crop_change = CropChangePredictor()
    # predictors = {"ELUC": nnp, "change": pct_change, "cropchange": crop_change}
    predictors = {"ELUC": nnp, "change": pct_change}

    for config_path in args.config_paths:
        print("Loading from config file: ", config_path)
        with open(Path(config_path), "r", encoding="utf-8") as f:
            config = json.load(f)

        base_seed_dir = config["evolution_params"].get("seed_dir", None)

        print("Initializing prescription...")
        for repeat in range(args.n_repeats):
            save_path = config["save_path"]
            if args.n_repeats > 1:
                print(f"Repeat {repeat + 1}/{args.n_repeats}")
                save_path = save_path + f"_{repeat}"

            # Seeding
            if "seed_dir" in config["evolution_params"].keys():
                seed_dir = config["evolution_params"]["seed_dir"]
                if args.n_repeats > 1:
                    seed_dir = base_seed_dir + f"_{repeat}"
                seed_dir = Path(seed_dir)
                if not seed_dir.exists():
                    print(f"Creating seeds in {config['evolution_params']['seed_dir']}...")
                    # Create seeds
                    seed_type = config["seed_type"]
                    if seed_type == "rhea":
                        seed_rhea(seed_dir)
                    elif seed_type == "base":
                        seed_base(seed_dir)
                    else:
                        raise ValueError("Invalid seed type")
                config["evolution_params"]["seed_dir"] = seed_dir

            tp = TorchTrainer(
                eval_df=dataset.train_df.sample(frac=0.001, random_state=42),
                encoder=encoder,
                predictors=predictors,
                batch_size=4096,
                **config["evolution_params"]
            )
            print("Training prescriptors...")
            final_pop = tp.neuroevolution(Path(save_path))
            print("Done!")
