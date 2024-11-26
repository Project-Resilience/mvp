from pathlib import Path

import pandas as pd


def get_gens_df(results_dir: Path, gens: list[int], pareto=False):
    """
    Gets the pareto df for multiple generations and merges them into one.
    :param dir: The experiment results directory.
    :param gens: List of generations to get the pareto df for.
    """
    dfs = []
    for gen in gens:
        gen_df = pd.read_csv(results_dir / f"{gen}.csv")
        if pareto:
            gen_df = gen_df[gen_df['NSGA-II_rank'] == 1]
        gen_df["gen"] = gen
        dfs.append(gen_df)
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def get_overall_pareto_df(final_gen: int, results_dir: Path, outcomes: list[str]):
    pareto_list = []
    
    all_pareto_df = get_gens_df(results_dir, list(range(1, final_gen + 1)), pareto=True)
    all_pareto_df = all_pareto_df.drop_duplicates(subset=["id"])

    for _, add in all_pareto_df.iterrows():
        pareto = True
        for _, compare in all_pareto_df.iterrows():
            dominated = False
            for outcome in outcomes:
                if add[outcome] < compare[outcome]:
                    dominated = False
                    break
                if add[outcome] > compare[outcome]:
                    dominated = True
            if dominated:
                pareto = False
                break
        if pareto:
            pareto_list.append(add)

    return pd.DataFrame(pareto_list)
