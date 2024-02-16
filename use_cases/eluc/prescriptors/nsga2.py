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
    while front[i] != []:
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

    del front[len(front)-1]

    # Convert front indices to candidates
    candidate_fronts = []
    for f in front:
        cands = []
        for idx in f:
            cands.append(candidates[idx])
        candidate_fronts.append(cands)

    return candidate_fronts, rank

def calculate_crowding_distance(front):
    """
    Calculate crowding distance of front.
    """
    n_objectives = len(front[0].metrics)
    distances = [0 for _ in range(len(front))]
    for m in range(n_objectives):
        front.sort(key=lambda c: c.metrics[m])
        obj_min = front[0].metrics[m]
        obj_max = front[-1].metrics[m]
        distances[0] = distances[-1] = float('inf')
        for i in range(1, len(front) - 1):
            distances[i] += (front[i+1].metrics[m] - front[i-1].metrics[m]) / (obj_max - obj_min)

    return distances

def dominates(candidate1, candidate2):
    """
    Determine if one individual dominates another.
    """
    for obj1, obj2 in zip(candidate1.metrics, candidate2.metrics):
        if obj1 > obj2:
            return False
    return True