from pathlib import Path

import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from prsdk.persistence.serializers.neural_network_serializer import NeuralNetSerializer

from data import constants
from data.eluc_data import ELUCData
from predictors.percent_change.percent_change_predictor import PercentChangePredictor
from prescriptors.pymoo.eluc_problem import ELUCProblem


def optimize():
    """
    Runs Pymoo optimization on the ELUC problem.
    """
    print("Loading data...")
    dataset = ELUCData.from_hf()
    eval_df = dataset.train_df.sample(frac=0.001, random_state=42)

    print("Loading predictors...")
    serializer = NeuralNetSerializer()
    predictors = [serializer.load(Path("predictors/trained_models/danyoung--eluc-global-nn")), PercentChangePredictor()]

    print("Setting up problem...")
    problem = ELUCProblem(eval_df,
                          {"in_size": len(constants.CAO_MAPPING["context"]),
                           "hidden_size": 16,
                           "out_size": len(constants.RECO_COLS)},
                          predictors,
                          batch_size=4096,
                          device="mps")
    
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    print("Optimizing...")
    res = minimize(problem, algorithm, get_termination("n_gen", 20), seed=42, save_history=True, verbose=True)
    return res


def main():
    """
    Main function that calls optimize and then plots the results.
    """
    res = optimize()
    X = res.X
    F = res.F
    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 1], F[:, 0], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.xlabel("Change")
    plt.ylabel("ELUC")
    plt.show()


if __name__ == "__main__":
    main()
