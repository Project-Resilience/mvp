"""
PyTorch implementation of NSGA-II.
"""
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
from prescriptors.prescriptor import Prescriptor
from prescriptors.nsga2 import nsga2_utils
from prescriptors.nsga2.candidate import Candidate

class TorchPrescriptor(Prescriptor):
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

        # We cache the training context here so that we don't have to repeatedly convert to tensor.
        # We can pass in our own dataframe later for inference.
        context_ds = TorchDataset(self.encoded_eval_df[constants.CAO_MAPPING["context"]].to_numpy(),
                                  np.zeros((len(self.encoded_eval_df), len(constants.RECO_COLS))))
        self.context_dl = torch.utils.data.DataLoader(context_ds, batch_size=batch_size, shuffle=False)
        self.batch_size = batch_size

        self.predictor = predictor

    def _reco_tensor_to_df(self, reco_tensor: torch.Tensor, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts raw Candidate neural network output tensor to scaled dataframe.
        Sets the indices of the recommendations so that we can subtract from the context to get
        the land diffs.
        """
        reco_df = pd.DataFrame(reco_tensor.cpu().numpy(), index=context_df.index, columns=constants.RECO_COLS)
        reco_df = reco_df.clip(0, None) # ReLU
        reco_df[reco_df.sum(axis=1) == 0] = 1 # Rows of all 0s are set to 1s
        reco_df = reco_df.div(reco_df.sum(axis=1), axis=0) # Normalize to sum to 1
        reco_df = reco_df.mul(context_df[constants.RECO_COLS].sum(axis=1), axis=0) # Rescale to match original sum
        return reco_df

    def _reco_to_context_actions(self, reco_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts recommendation df and original context df to context + actions df.
        Uses original context to compute diffs based on recommendations - original context.
        """
        assert reco_df.index.isin(context_df.index).all(), "Recommendation index must be a subset of context index."
        presc_actions_df = reco_df - context_df[constants.RECO_COLS]
        presc_actions_df = presc_actions_df.rename(constants.RECO_MAP, axis=1)
        presc_actions_df[constants.NO_CHANGE_COLS] = 0
        context_actions_df = pd.concat([context_df[constants.CAO_MAPPING["context"]],
                                            presc_actions_df[constants.CAO_MAPPING["actions"]]],
                                            axis=1)
        return context_actions_df

    def _prescribe(self, candidate: Candidate, context_df=None) -> pd.DataFrame:
        """
        Prescribes actions given a candidate and a context.
        If we don't provide a context_df, we use the stored context_dl to avoid overhead. 
        Otherwise, we create a new dataloader from the given context_df.
        Overall flow of prescription:
            1. context_df -> context_tensor
            2. candidate(context_tensor) -> reco_tensor
            3. reco_tensor -> reco_df
            4. context_df, reco_df -> context_actions_df
        """
        # Either create context_dl or used stored one
        context_dl = None
        if context_df is not None:
            encoded_context_df = self.encoder.encode_as_df(context_df[constants.CAO_MAPPING["context"]])
            context_ds = TorchDataset(encoded_context_df.to_numpy(),
                                      np.zeros((len(encoded_context_df), len(constants.RECO_COLS))))
            context_dl = torch.utils.data.DataLoader(context_ds, batch_size=self.batch_size, shuffle=False)
        else:
            context_df = self.eval_df
            context_dl = self.context_dl

        # Aggregate recommendations
        reco_list = []
        with torch.no_grad():
            for X, _ in context_dl:
                recos = candidate(X)
                reco_list.append(recos)
            reco_tensor = torch.concatenate(reco_list, dim=0)

            # Convert recommendations into context + actions
            reco_df = self._reco_tensor_to_df(reco_tensor, context_df)

        context_actions_df = self._reco_to_context_actions(reco_df, context_df)
        return context_actions_df

    def predict_metrics(self, context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes ELUC and change for each sample in a context_actions_df.
        """
        eluc_df = self.predictor.predict(context_actions_df)
        change_df = self.compute_percent_changed(context_actions_df)

        return eluc_df, change_df

    def _evaluate_candidates(self, candidates: list[Candidate]):
        """
        Calls prescribe and predict on candidates and assigns their metrics to the results.
        """
        for candidate in candidates:
            context_actions_df = self._prescribe(candidate)
            eluc_df, change_df = self.predict_metrics(context_actions_df)
            candidate.metrics = (eluc_df["ELUC"].mean(), change_df["change"].mean())

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
            crowding_distance = nsga2_utils.calculate_crowding_distance(front)
            for candidate, distance in zip(front, crowding_distance):
                candidate.distance = distance
            if len(parents) + len(front) > n_parents:  # If adding this front exceeds num_parents
                front = sorted(front, key=lambda cand: cand.distance, reverse=True)
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

    def _make_new_pop(self, parents: list[Candidate], pop_size: int, gen:int) -> list[Candidate]:
        """
        Makes new population by creating children from parents.
        We use tournament selection to select parents for crossover.
        """
        sorted_parents = sorted(parents, key=lambda cand: (cand.rank, -cand.distance))
        children = []
        for i in range(pop_size):
            parent1, parent2 = self._tournament_selection(sorted_parents)
            child = Candidate.from_crossover(parent1, parent2, self.p_mutation, gen, i)
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
            shutil.rmtree(save_path)
        save_path.mkdir(parents=True, exist_ok=False)
        results = []
        parents = [Candidate(**self.candidate_params, gen=1, cand_id=i) for i in range(self.pop_size)]
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
        gen_results = [cand.record_state() for cand in candidates]
        gen_results_df = pd.DataFrame(gen_results)
        gen_results_df.to_csv(save_path / f"{gen}.csv", index=False)

        # Save rank 1 candidate state dicts
        (save_path / f"{gen}").mkdir(parents=True, exist_ok=True)
        pareto_candidates = [cand for cand in candidates if cand.rank == 1]
        for cand in pareto_candidates:
            torch.save(cand.state_dict(), save_path / f"{gen}" / f"{cand.gen}_{cand.cand_id}.pt")

    def _record_candidate_avgs(self, gen: int, candidates: list[Candidate]) -> dict:
        """
        Gets the average eluc and change for a population of candidates.
        """
        avg_eluc = np.mean([c.metrics[0] for c in candidates])
        avg_change = np.mean([c.metrics[1] for c in candidates])
        return {"gen": gen, "eluc": avg_eluc, "change": avg_change}

    def prescribe_land_use(self, context_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Wrapper for prescribe method that loads a candidate from disk using an id.
        Valid kwargs:
            cand_id: str, the ID of the candidate to load
            results_dir: Path, the directory where the candidate is stored
        Then takes in a context dataframe and prescribes actions.
        """
        candidate = Candidate(**self.candidate_params)
        gen = int(kwargs["cand_id"].split("_")[0])
        state_dict = torch.load(kwargs["results_dir"] / f"{gen + 1}" / f"{kwargs['cand_id']}.pt")
        candidate.load_state_dict(state_dict)

        context_actions_df = self._prescribe(candidate, context_df)
        return context_actions_df
    