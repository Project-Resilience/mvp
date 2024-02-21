import torch

class Candidate(torch.nn.Module):
    """
    Simple fixed topology 1 hidden layer feed-forward nn candidate
    """
    def __init__(self, in_size: int, hidden_size: int, out_size: int, device="cpu", gen=-1, cand_id=-1, parents=(None, None)):
        super().__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, out_size))
        
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Orthogonal initialization
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight)
                layer.bias.data.fill_(0.01)

        # To keep track of metrics
        self.metrics = None
        self.rank = None
        self.distance = -1

        # For evolution logging purposes
        self.gen = gen
        self.cand_id = cand_id
        self.parents = parents

    @classmethod
    def from_crossover(cls, parent1, parent2, p_mutation: float, gen: int, cand_id: int):
        """
        Crossover two parents to create a child.
        Take a random 50/50 choice of either parent's weights
        """
        child = cls(in_size=parent1.in_size,
                    hidden_size=parent1.hidden_size,
                    out_size=parent1.out_size,
                    device=parent1.device,
                    gen=gen,
                    cand_id=cand_id,
                    parents=(parent1.cand_id, parent2.cand_id))
        
        for child_param, parent1_param, parent2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
            mask = torch.rand(size=child_param.data.shape) < 0.5
            child_param.data = torch.where(mask, parent1_param.data, parent2_param.data)
        child.mutate(p_mutation)
        return child
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the simple nn
        """
        out = self.model(X)
        return out
    
    def mutate(self, p_mutation: float):
        """
        Randomly mutates each weight with probability p_mutation with gaussian noise mu=0, sigma=0.1
        """
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                mask = torch.rand(size=layer.weight.shape) < p_mutation
                layer.weight.data += mask * torch.randn(size=layer.weight.shape) * 0.1
                mask = torch.rand(size=layer.bias.shape) < p_mutation
                layer.bias.data += mask * torch.randn(size=layer.bias.shape) * 0.1