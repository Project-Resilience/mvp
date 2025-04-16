"""
Contains the Candidate class, which is a simple feed-forward neural network that keeps track
of its own metrics and logging information.
"""
import torch


class Candidate(torch.nn.Module):
    """
    Simple fixed topology 1 hidden layer feed-forward nn candidate.
    Keeps track of its own metrics and evolution logging information.
    """
    def __init__(self, in_size: int, hidden_size: int, out_size: int,
                 device="cpu", cand_id="-1", parents=(None, None)):
        super().__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size))

        # Orthogonal initialization
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)

        self.device = device
        self.model.to(device)
        self.model.eval()

        # To keep track of metrics
        self.metrics = ()
        self.rank = None
        self.distance = -1

        # For evolution logging purposes
        self.cand_id = cand_id
        self.parents = parents

    @classmethod
    def from_crossover(cls,
                       parent1: "Candidate",
                       parent2: "Candidate",
                       p_mutation: float,
                       mutation_factor: float,
                       cand_id: str) -> "Candidate":
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights
        """
        child = cls(in_size=parent1.in_size,
                    hidden_size=parent1.hidden_size,
                    out_size=parent1.out_size,
                    device=parent1.device,
                    cand_id=cand_id,
                    parents=(parent1.cand_id, parent2.cand_id))

        params = zip(child.parameters(), parent1.parameters(), parent2.parameters())
        for child_param, parent1_param, parent2_param in params:
            mask = torch.rand(size=child_param.data.shape) < 0.5
            child_param.data = torch.where(mask, parent1_param.data, parent2_param.data)
        child.mutate(p_mutation, mutation_factor)
        return child

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the simple nn
        """
        out = self.model(X)
        return out

    def mutate(self, p_mutation: float, mutation_factor: float):
        """
        Randomly mutates each weight with probability p_mutation with gaussian noise mu=0, sigma=0.1
        """
        with torch.no_grad():
            for param in self.model.parameters():
                mutate_mask = torch.rand(param.shape, device=param.device) < p_mutation
                noise = torch.normal(0,
                                     mutation_factor,
                                     param[mutate_mask].shape,
                                     device=param.device,
                                     dtype=param.dtype)
                param[mutate_mask] += noise * param[mutate_mask]

    def record_state(self) -> dict:
        """
        Record the state of the candidate for logging purposes
        """
        if len(self.metrics) == 0:
            raise ValueError("Candidate has not been evaluated yet")
        cand_state = {"id": self.cand_id,
                      "parents": self.parents,
                      "NSGA-II_rank": self.rank,  # Named this to match ESP
                      "distance": self.distance}

        cand_state["ELUC"] = self.metrics[0]
        cand_state["change"] = self.metrics[1]
        if len(self.metrics) > 2:
            cand_state["cropchange"] = self.metrics[2]

        return cand_state
