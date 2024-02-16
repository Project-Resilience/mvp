from pathlib import Path

from data import constants
from data.eluc_data import ELUCData
from prescriptors.nsga2.torch_prescriptor import TorchPrescriptor
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor

if __name__ == "__main__":
    dataset = ELUCData(start_year=2020, test_year=2021, countries=["US"])
    nnp = NeuralNetPredictor()
    nnp.load("predictors/neural_network/trained_models/experiment_nn")
    tp = TorchPrescriptor(
        pop_size=100,
        n_generations=100,
        n_elites=10,
        p_mutation=0.2,
        eval_df=dataset.train_df.sample(frac=0.001, random_state=42),
        encoder=dataset.encoder,
        predictor=nnp,
        batch_size=4096,
        in_size=len(constants.CAO_MAPPING["context"]),
        hidden_size=16,
        out_size=len(constants.RECO_COLS)
    )

    save_path = Path("prescriptors/nsga2/trained_prescriptors/test")
    final_pop = tp.neuroevolution(save_path)