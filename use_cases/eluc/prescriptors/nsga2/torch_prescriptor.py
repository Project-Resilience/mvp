import random
import shutil
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from data import constants
from data.eluc_data import ELUCEncoder
from data.torch_data import TorchDataset
from predictors.predictor import Predictor
from prescriptors.nsga2 import nsga2_utils
from prescriptors.nsga2.candidate import Candidate

class TorchPrescriptor():
    """
    Handles prescriptor candidate evolution
    """
    def __init__(self,
                 pop_size: int,
                 n_generations: int,
                 p_mutation: float,
                 eval_df: pd.DataFrame,
                 encoder: ELUCEncoder,
                 predictor: Predictor,
                 batch_size: int,
                 candidate_params: dict,
                 seed_dir=None):
        
        self.candidate_params = candidate_params
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.p_mutation = p_mutation
        self.seed_dir=seed_dir

        self.eval_df = eval_df
        self.encoded_eval_df = encoder.encode_as_df(eval_df)
        self.encoder = encoder
        context_ds = TorchDataset(self.encoded_eval_df[constants.CAO_MAPPING["context"]].to_numpy(),
                                  np.zeros((len(self.encoded_eval_df), len(constants.RECO_COLS))))
        self.context_dl = torch.utils.data.DataLoader(context_ds, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size

        self.predictor = predictor

    def _reco_tensor_to_df(self, reco_tensor: torch.Tensor) -> pd.DataFrame:
        """
        Converts neural network output tensor to scaled dataframe.
        """
        reco_df = pd.DataFrame(reco_tensor.cpu().numpy(), index=self.eval_df.index, columns=constants.RECO_COLS)
        reco_df = reco_df.clip(0, None) # ReLU
        reco_df[reco_df.sum(axis=1) == 0] = 1 # Rows of all 0s are set to 1s
        reco_df = reco_df.div(reco_df.sum(axis=1), axis=0) # Normalize to sum to 1
        reco_df = reco_df.mul(self.eval_df[constants.RECO_COLS].sum(axis=1), axis=0) # Rescale to match original sum
        return reco_df
    
    def _reco_to_context_actions(self, reco_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts recommendation dataframe to context + actions dataframe.
        """
        presc_actions_df = reco_df - self.eval_df[constants.RECO_COLS]
        presc_actions_df = presc_actions_df.rename(constants.RECO_MAP, axis=1)
        presc_actions_df[constants.NO_CHANGE_COLS] = 0
        context_actions_df = pd.concat([self.eval_df[constants.CAO_MAPPING["context"]],
                                            presc_actions_df[constants.CAO_MAPPING["actions"]]],
                                            axis=1)
        return context_actions_df

    def _compute_percent_changed(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates percent of land changed by prescriptor.
        """
        # Sum the positive diffs
        percent_changed = context_actions_df[context_actions_df[constants.DIFF_LAND_USE_COLS] > 0][constants.DIFF_LAND_USE_COLS].sum(axis=1)
        # Divide by sum of used land
        percent_changed = percent_changed / context_actions_df[constants.LAND_USE_COLS].sum(axis=1)
        change_df = pd.DataFrame(percent_changed, columns=["change"])
        return change_df
    
    def prescribe_and_predict(self, candidate: Candidate) -> tuple:
        """
        Takes a candidate and prescribes actions, then predicts metrics.
        """
        # Aggregate recommendations
        reco_list = []
        with torch.no_grad():
            for X, _ in self.context_dl:
                recos = candidate(X)
                reco_list.append(recos)
            reco_tensor = torch.concatenate(reco_list, dim=0)

            # Convert reccomendations into context + actions
            reco_df = self._reco_tensor_to_df(reco_tensor)
        
        context_actions_df = self._reco_to_context_actions(reco_df)
        # Compute metrics
        eluc_df = self.predictor.predict(context_actions_df)
        change_df = self._compute_percent_changed(context_actions_df)
        
        return eluc_df["ELUC"].mean(), change_df["change"].mean()

    def evaluate_candidates(self, candidates: list):
        """
        Calls prescribe and predict on candidates and assigns their metrics to the results.
        """
        for candidate in candidates:
            eluc, change = self.prescribe_and_predict(candidate)
            candidate.metrics = (eluc, change)

    def select_parents(self, candidates: list, n_parents: int):
        """
        NSGA-II parent selection using fast non-dominated sort and crowding distance.
        """
        fronts, ranks = nsga2_utils.fast_non_dominated_sort(candidates)
        for candidate, rank in zip(candidates, ranks):
            candidate.rank = rank
        parents = []
        for front in fronts:
            # Compute crowding distance here even though it's technically not necessary now
            # so that later we can sort by distance
            crowding_distance = nsga2_utils.calculate_crowding_distance(front)
            for candidate, distance in zip(front, crowding_distance):
                candidate.distance = distance
            if len(parents) + len(front) > n_parents:  # If adding this front exceeds num_parents
                front = sorted(front, key=lambda c: c.distance, reverse=True)
                parents += front[:n_parents - len(parents)]
                break
            parents += front
        return parents
    
    def tournament_selection(self, sorted_parents: list) -> tuple:
        """
        Same implementation as in ESP.
        Takes two random parents and compares their indices since this is a measure of their performance.
        Note: It is possible for this function to select the same parent twice.
        """
        idx1 = min(random.choices(range(len(sorted_parents)), k=2))
        idx2 = min(random.choices(range(len(sorted_parents)), k=2))
        return sorted_parents[idx1], sorted_parents[idx2]

    def make_new_pop(self, parents: list, pop_size: int, gen:int) -> list:
        """
        Makes new population by creating children from parents.
        We use tournament selection to select parents for crossover.
        """
        sorted_parents = sorted(parents, key=lambda c: (c.rank, -c.distance))
        children = []
        for i in range(pop_size):
            parent1, parent2 = self.tournament_selection(sorted_parents)
            child = Candidate.from_crossover(parent1, parent2, self.p_mutation, gen, i)
            children.append(child)
        return children

    def neuroevolution(self, save_path: Path):
        """
        Main Neuroevolution Loop that performs NSGA-II.
        After initializing the first population randomly, goes through 3 steps in each generation:
        1. Evaluate candidates
        2. Select parents
        3. Make new population from parents
        """
        shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=False)
        results = []
        parents = [Candidate(**self.candidate_params, gen=1, cand_id=i) for i in range(self.pop_size)]
        # Seeding the first generation with trained models
        if self.seed_dir:
            seed_paths = list(self.seed_dir.glob("*.pt"))
            for i, seed_path in enumerate(seed_paths):
                print(f"Seeding with {seed_path}...")
                parents[i].load_state_dict(torch.load(seed_path))

        offspring = []
        for gen in tqdm(range(1, self.n_generations+1)):
            # Set up candidates by merging parent and offspring populations
            candidates = parents + offspring
            self.evaluate_candidates(candidates)

            # NSGA-II parent selection
            parents = self.select_parents(candidates, self.pop_size)

            # Record the performance of the most successful candidates
            results.append(self._record_candidate_avgs(gen+1, parents))
            self._record_gen_results(gen, parents, save_path)

            # If we aren't on the last generation, make a new population
            if gen < self.n_generations:
                offspring = self.make_new_pop(parents, self.pop_size, gen)

        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path / "results.csv", index=False)

        return parents
    
    def _record_gen_results(self, gen: int, candidates: list, save_path: Path):
        # Save statistics of candidates
        gen_results = [c.record_state() for c in candidates]
        gen_results_df = pd.DataFrame(gen_results)
        gen_results_df.to_csv(save_path / f"{gen}.csv", index=False)

        # Save rank 1 candidate state dicts
        (save_path / f"{gen}").mkdir(parents=True, exist_ok=True)
        pareto_candidates = [c for c in candidates if c.rank == 1]
        for c in pareto_candidates:
            torch.save(c.state_dict(), save_path / f"{gen}" / f"{c.gen}_{c.cand_id}.pt")

    def _record_candidate_avgs(self, gen, candidates):
        avg_eluc = np.mean([c.metrics[0] for c in candidates])
        avg_change = np.mean([c.metrics[1] for c in candidates])
        return {"gen": gen, "eluc": avg_eluc, "change": avg_change}
    