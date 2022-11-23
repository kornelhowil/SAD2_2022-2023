import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(x_dim, h_dim1)
        self.lin2 = nn.Linear(h_dim1, h_dim2)
        self.lin31 = nn.Linear(h_dim2, z_dim)  # mu
        self.lin32 = nn.Linear(h_dim2, z_dim)  # log_var
        
    def forward(self, x):
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        return self.lin31(x), self.lin32(x)  # mu, log_var
