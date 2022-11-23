import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Decoder, self).__init__()
        self.lin4 = nn.Linear(z_dim, h_dim2)
        self.lin5 = nn.Linear(h_dim2, h_dim1)
        self.lin6 = nn.Linear(h_dim1, x_dim)

    @staticmethod
    def log_prob(x, p):
        distribution = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs=p)
        return -torch.sum(distribution.log_prob(x))

    def forward(self, z):
        h = nn.functional.relu(self.lin4(z))
        h = nn.functional.relu(self.lin5(h))
        return torch.sigmoid(self.lin6(h))
