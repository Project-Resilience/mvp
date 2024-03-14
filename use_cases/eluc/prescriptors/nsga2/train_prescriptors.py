import argparse
from pathlib import Path

from data import constants
from data.eluc_data import ELUCData
from prescriptors.nsga2.torch_prescriptor import TorchPrescriptor
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor

if __name__ == "__main__":

    print("Loading dataset...")
    dataset = ELUCData()
    print("Loading predictor...")
    nnp = NeuralNetPredictor(device="cuda")
    nnp.load("predictors/neural_network/trained_models/no_overlap_nn")
    print("Initializing prescription...")
    candidate_params = {"in_size": len(constants.CAO_MAPPING["context"]),
                        "hidden_size": 16,
                        "out_size": len(constants.RECO_COLS)}
    tp = TorchPrescriptor(
        pop_size=100,
        n_generations=100,
        p_mutation=0.2,
        eval_df=dataset.train_df.sample(frac=0.001, random_state=42),
        encoder=dataset.encoder,
        predictor=nnp,
        batch_size=4096,
        candidate_params=candidate_params,
        seed_dir=Path("prescriptors/nsga2/seeds/test")
    )
    print("Training prescriptors...")
    save_path = Path("prescriptors/nsga2/trained_prescriptors/test")
    final_pop = tp.neuroevolution(save_path)
    print("Done!")