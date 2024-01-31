import torch

class ELUCNeuralNet(torch.nn.Module):
    """
    Custom torch neural network module.
    :param in_size: number of input features
    :param hidden_sizes: list of hidden layer sizes
    :param linear_skip: whether to concatenate input to hidden layer output
    :param dropout: dropout probability
    """
    class EncBlock(torch.nn.Module):
        def __init__(self, in_size, out_size, dropout):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(in_size, out_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout)
            )
        def forward(self, X):
            return self.model(X)

    def __init__(self, in_size, hidden_sizes, linear_skip, dropout):
        super().__init__()
        self.linear_skip = linear_skip
        hidden_sizes = [in_size] + hidden_sizes
        enc_blocks = [self.EncBlock(hidden_sizes[i], hidden_sizes[i+1], dropout) for i in range(len(hidden_sizes) - 1)]
        self.enc = torch.nn.Sequential(*enc_blocks)
        out_size = hidden_sizes[-1] + in_size if linear_skip else hidden_sizes[-1]
        self.linear = torch.nn.Linear(out_size, 1)

    def forward(self, X):
        hid = self.enc(X)
        if self.linear_skip:
            hid = torch.concatenate([hid, X], dim=1)
        out = self.linear(hid)
        return out