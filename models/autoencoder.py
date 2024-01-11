import torch
import torch.nn as nn

class AutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim: int, features: int = 512):

        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.features = features

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=self.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=self.features),
            nn.Sigmoid()
        )

    def decode(self, encoded: torch.Tensor):
        return self.decoder(encoded)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded


class WassersteinAutoEncoder(nn.Module):

    def __init__(self, latent_dim: int, features: int = 512):

        super(WassersteinAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.features = features

        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.features)
        )

    def decode(self, encoded: torch.Tensor):
        return self.decoder(encoded)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

    def mmd_loss(self, y: torch.Tensor, sigma: float):
        z = torch.randn_like(y)

        yy = torch.sum(sigma / (sigma + torch.cdist(y, y, p=2) ** 2))
        zz = torch.sum(sigma / (sigma + torch.cdist(z, z, p=2) ** 2))
        yz = torch.sum(sigma / (sigma + torch.cdist(y, z, p=2) ** 2))
        div_1_n_sq = 1 / y.size(0) ** 2

        return div_1_n_sq * (yy + zz) - 2 * div_1_n_sq * yz