import torch
import torch.nn as nn


class CustomDecoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, h_dim3, z_dim):
        super(CustomDecoder, self).__init__()
        self.lin5 = nn.Linear(z_dim, h_dim3)
        self.batch_norm5 = nn.BatchNorm1d(h_dim3)
        self.drop5 = nn.Dropout()
        self.lin6 = nn.Linear(h_dim3, h_dim2)
        self.drop6 = nn.Dropout()
        self.batch_norm6 = nn.BatchNorm1d(h_dim2)
        self.lin7 = nn.Linear(h_dim2, h_dim1)
        self.drop7 = nn.Dropout()
        self.batch_norm7 = nn.BatchNorm1d(h_dim1)
        self.lin8 = nn.Linear(h_dim1, x_dim)

    @staticmethod
    def log_prob(x, p):
        dist = torch.distributions.poisson.Poisson(p + 1e-4)
        log_prob_tensor = dist.log_prob(x)
        log_prob_sum = torch.sum(log_prob_tensor)
        return log_prob_sum

    def forward(self, z):
        h = self.drop5(self.batch_norm5(self.lin5(z)))
        h = nn.functional.relu(h)
        h = self.drop6(self.batch_norm6(self.lin6(h)))
        h = nn.functional.relu(h)
        h = self.drop7(self.batch_norm7(self.lin7(h)))
        h = nn.functional.relu(h)
        return nn.functional.relu(self.lin8(h))