import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def sampling(mu, log_var):
        std = torch.sqrt(torch.exp(log_var))
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def kl_div(mu, sigma):
        n = mu.shape[1]
        mu = mu.reshape(mu.shape[0], mu.shape[1], 1)
        tr_term = torch.sum(sigma, dim=1)
        det_term = -torch.log(torch.prod(sigma, dim=1))
        quad_term = torch.transpose(mu, 1, 2) @ mu
        return torch.sum(0.5 * (tr_term + det_term + quad_term - n))

    def loss_function(self, data, beta):
        p, mu, log_var = self.forward(data)
        kl_loss = self.kl_div(mu, torch.exp(log_var))
        recon_loss = self.decoder.log_prob(torch.round(data), p)
        return recon_loss + beta * kl_loss

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        mu_d = self.decoder(z)
        return mu_d, mu, log_var

    def generate(self, z):
        return self.decoder(z)
