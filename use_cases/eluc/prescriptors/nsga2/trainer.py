"""
PyTorch implementation of NSGA-II.
"""
import random
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from prsdk.data.torch_data import TorchDataset
from prsdk.predictors.predictor import Predictor

from data import constants
from data.eluc_data import ELUCEncoder
from prescriptors.nsga2 import nsga2_utils
from prescriptors.nsga2.candidate import Candidate
from prescriptors.nsga2.land_use_prescriptor import LandUsePrescriptor
from prescriptors.prescriptor_manager import PrescriptorManager


class TorchTrainer():
    """
    Handles prescriptor candidate evolution
    """
    def __init__(self,
                 pop_size: int,
                 n_generations: int,
                 p_mutation: float,
                 eval_df: pd.DataFrame,
                 encoder: ELUCEncoder,
                 predictors: dict[str, Predictor],
                 batch_size: int,
                 candidate_params: dict,
                 seed_dir=None):

        # Evolution params
        self.candidate_params = candidate_params
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.p_mutation = p_mutation
        self.seed_dir = seed_dir

        # Evaluation params
        self.encoder = encoder
        self.predictors = predictors
        self.context_df = eval_df[constants.CAO_MAPPING["context"]]
        encoded_eval_df = encoder.encode_as_df(eval_df)
        context_ds = TorchDataset(encoded_eval_df[constants.CAO_MAPPING["context"]].to_numpy(),
                                  np.zeros((len(encoded_eval_df), len(constants.RECO_COLS))))
        self.encoded_context_dl = DataLoader(context_ds, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size

    def _evaluate_candidates(self, candidates: list[Candidate]):
        """
        Calls prescribe and predict on candidates and assigns their metrics to the results.
        This is where the Project Resilience Prescriptor logic is used in evolution, although it doesn't have to be.
        We wrap a LandUsePrescriptor around the Candidate we are evaluating and call torch_prescribe which is a
        special prescription method that goes straight from tensor to tensor instead of converting to DataFrame.
        We use a dummy PrescriptorManager to compute the metrics using predict_metrics.
        """
        prescriptor_manager = PrescriptorManager(None, self.predictors)
        for candidate in candidates:
            prescriptor = LandUsePrescriptor(candidate, self.encoder, self.batch_size)
            context_actions_df = prescriptor.torch_prescribe(self.context_df, self.encoded_context_dl)
            outcomes_df = prescriptor_manager.predict_metrics(context_actions_df)
            candidate.metrics = (outcomes_df["ELUC"].mean(),
                                 outcomes_df["change"].mean(),
                                 outcomes_df["cropchange"].mean())

    def _select_parents(self, candidates: list[Candidate], n_parents: int) -> list[Candidate]:
        """
        NSGA-II parent selection using fast non-dominated sort and crowding distance.
        Sets candidates' ranks and distance attributes.
        """
        fronts, ranks = nsga2_utils.fast_non_dominated_sort(candidates)
        for candidate, rank in zip(candidates, ranks):
            candidate.rank = rank
        parents = []
        for front in fronts:
            # Compute crowding distance here even though it's technically not necessary now
            # so that later we can sort by distance
            nsga2_utils.calculate_crowding_distance(front)
            if len(parents) + len(front) > n_parents:  # If adding this front exceeds num_parents
                front = sorted(front, key=lambda candidate: candidate.distance, reverse=True)
                parents += front[:n_parents - len(parents)]
                break
            parents += front
        return parents

    def _tournament_selection(self, sorted_parents: list[Candidate]) -> tuple[Candidate, Candidate]:
        """
        Takes two random parents and compares their indices since this is a measure of their performance.
        Note: It is possible for this function to select the same parent twice.
        """
        idx1 = min(random.choices(range(len(sorted_parents)), k=2))
        idx2 = min(random.choices(range(len(sorted_parents)), k=2))
        return sorted_parents[idx1], sorted_parents[idx2]

    def _make_new_pop(self, parents: list[Candidate], pop_size: int, gen: int) -> list[Candidate]:
        """
        Makes new population by creating children from parents.
        We use tournament selection to select parents for crossover.
        """
        sorted_parents = sorted(parents, key=lambda candidate: (candidate.rank, -candidate.distance))
        children = []
        for i in range(pop_size):
            parent1, parent2 = self._tournament_selection(sorted_parents)
            child = Candidate.from_crossover(parent1, parent2, self.p_mutation, f"{gen}_{i}")
            children.append(child)
        return children

    def neuroevolution(self, save_path: Path):
        """
        Main Neuroevolution Loop that performs NSGA-II.
        After initializing the first population randomly, goes through 3 steps in each generation:
        1. Evaluate candidates
        2. Select parents
        2a Log performance of parents
        3. Make new population from parents
        """
        if save_path.exists():
            raise ValueError(f"Path {save_path} already exists. Please choose a new path.")
        save_path.mkdir(parents=True, exist_ok=False)
        print(f"Saving to {save_path}")
        self.encoder.save_fields(save_path / "fields.json")
        results = []
        parents = [Candidate(**self.candidate_params, cand_id=f"1_{i}") for i in range(self.pop_size)]
        # Seeding the first generation with trained models
        if self.seed_dir:
            seed_paths = list(self.seed_dir.glob("*.pt"))
            for idx, seed_path in enumerate(seed_paths):
                print(f"Seeding with {seed_path}...")
                parents[idx].load_state_dict(torch.load(seed_path))

        offspring = []
        for gen in tqdm(range(1, self.n_generations+1)):
            # Set up candidates by merging parent and offspring populations
            candidates = parents + offspring
            self._evaluate_candidates(candidates)

            # NSGA-II parent selection
            parents = self._select_parents(candidates, self.pop_size)

            # Record the performance of the most successful candidates
            results.append(self._record_candidate_avgs(gen+1, parents))
            self._record_gen_results(gen, parents, save_path)

            # If we aren't on the last generation, make a new population
            if gen < self.n_generations:
                offspring = self._make_new_pop(parents, self.pop_size, gen)

        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path / "results.csv", index=False)

        return parents

    def _record_gen_results(self, gen: int, candidates: list[Candidate], save_path: Path) -> None:
        """
        Records the state of all the candidates.
        Save the pareto front to disk.
        """
        # Save statistics of candidates
        gen_results = [candidate.record_state() for candidate in candidates]
        gen_results_df = pd.DataFrame(gen_results)
        gen_results_df.to_csv(save_path / f"{gen}.csv", index=False)

        # Save rank 1 candidate state dicts from this generation
        (save_path / f"{gen}").mkdir(parents=True, exist_ok=True)
        this_gen_candidates = [candidate for candidate in candidates if candidate.cand_id.startswith(f"{gen}")]
        pareto_candidates = [candidate for candidate in this_gen_candidates if candidate.rank == 1]
        for candidate in pareto_candidates:
            torch.save(candidate.state_dict(), save_path / f"{gen}" / f"{candidate.cand_id}.pt")

    def _record_candidate_avgs(self, gen: int, candidates: list[Candidate]) -> dict:
        """
        Gets the average eluc and change for a population of candidates.
        """
        avg_eluc = np.mean([c.metrics[0] for c in candidates])
        avg_change = np.mean([c.metrics[1] for c in candidates])
        return {"gen": gen, "eluc": avg_eluc, "change": avg_change}
