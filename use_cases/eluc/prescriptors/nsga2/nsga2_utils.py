"""
Utilities for NSGA-II implementation.
"""
from prescriptors.nsga2.candidate import Candidate


# pylint: disable=invalid-name
def fast_non_dominated_sort(candidates: list):
    """
    Fast non-dominated sort algorithm from ChatGPT
    """
    population_size = len(candidates)
    S = [[] for _ in range(population_size)]
    front = [[]]
    n = [0 for _ in range(population_size)]
    rank = [0 for _ in range(population_size)]

    for p in range(population_size):
        S[p] = []
        n[p] = 0
        for q in range(population_size):
            if dominates(candidates[p], candidates[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif dominates(candidates[q], candidates[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    # With this implementation the final front will be empty
    del front[len(front)-1]

    # Convert front indices to candidates
    candidate_fronts = []
    for f in front:
        cands = []
        for idx in f:
            cands.append(candidates[idx])
        candidate_fronts.append(cands)

    # Manually increment all ranks by 1 to match NSGA-II convention
    rank = [r + 1 for r in rank]

    return candidate_fronts, rank


def calculate_crowding_distance(front):
    """
    Set crowding distance of each candidate in front.
    """
    n_objectives = len(front[0].metrics)
    for candidate in front:
        candidate.distance = 0
    for m in range(n_objectives):
        sorted_front = sorted(front, key=lambda candidate: candidate.metrics[m])
        obj_min = sorted_front[0].metrics[m]
        obj_max = sorted_front[-1].metrics[m]
        sorted_front[0].distance = float('inf')
        sorted_front[-1].distance = float('inf')
        for i in range(1, len(sorted_front) - 1):
            if obj_max != obj_min:
                dist = sorted_front[i+1].metrics[m] - sorted_front[i-1].metrics[m]
                sorted_front[i].distance += dist / (obj_max - obj_min)
            # If all candidates have the same value, their distances are 0
            else:
                sorted_front[i].distance += 0


def dominates(candidate1: Candidate, candidate2: Candidate) -> bool:
    """
    Determine if one individual dominates another.
    One individual dominates another if it's doing better in at least one objective
    and better than or equal to in all the rest.
    Metrics are always minimized: lower is better.
    """
    better = False
    for obj1, obj2 in zip(candidate1.metrics, candidate2.metrics):
        if obj1 > obj2:
            return False
        if obj1 < obj2:
            better = True
    return better
