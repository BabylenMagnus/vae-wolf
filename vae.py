import torch
import torch.nn as nn
import numpy as np


def layers_encoder(n: int):
    standart_set = {'kernel_size': 3,
                    'stride': 2,
                    'padding': 1}

    layers = [nn.Conv2d(3, 8, **standart_set),
              nn.BatchNorm2d(8),
              nn.LeakyReLU(),
              nn.Dropout2d(p=0.25)]

    channel = 8

    for _ in range(int(np.log2(n) - 4)):
        layers.append(nn.Conv2d(channel, channel * 2, **standart_set))
        channel = channel * 2
        layers.append(nn.BatchNorm2d(channel))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout2d(p=0.25))

    return layers


class Encoder(nn.Module):

    def __init__(self, n: int, device):
        super(Encoder, self).__init__()

        self.device = device

        self.layers = layers_encoder(n)

        self.main = nn.Sequential(*self.layers)

        self.x = np.log2(n) - 4

        self.mu = nn.Linear(int(2 ** (self.x + 9)), 200)
        self.log_var = nn.Linear(int(2 ** (self.x + 9)), 200)

    def sampling(self, mu, log_var):
        epsilon = torch.from_numpy(np.random.normal(0., 1., 200))

        epsilon = epsilon.to(self.device)

        out = mu + torch.exp(log_var / 2) * epsilon
        return out

    def forward(self, img):
        img = self.main(img)

        img = img.view(-1, int(2 ** (self.x + 9)))

        mu = self.mu(img)
        log_var = self.log_var(img)

        img = self.sampling(mu, log_var)
        return img.float()


def layers_Decoder(n: int):
    standart_set = {'kernel_size': 4,
                    'stride': 2,
                    'padding': 1}

    layers = []

    n_ = n
    for _ in range(int(np.log2(n_) - 3)):
        layers.append(nn.ConvTranspose2d(int(n), int(n / 2), **standart_set))
        n = int(n / 2)
        layers.append(nn.BatchNorm2d(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout2d(p=0.25))

    return layers


class Decoder(nn.Module):

    def __init__(self, n: int):
        super(Decoder, self).__init__()

        self.x = np.log2(n) - 4

        self.first = nn.Linear(200, int(2 ** (9 + self.x)))

        self.main = nn.Sequential(*layers_Decoder(2 ** (3 + self.x)))

        self.final = nn.Sequential(nn.ConvTranspose2d(8, 3,
                                                      kernel_size=4,
                                                      stride=2,
                                                      padding=1),
                                   nn.Sigmoid())

    def forward(self, vector):
        vector = self.first(vector)
        vector = vector.view(-1, int(2 ** (3 + self.x)), 8, 8)

        img = self.main(vector)
        img = self.final(img)
        return img


class VAE(nn.Module):

    def __init__(self, n, device):
        super(VAE, self).__init__()

        self.device = device

        self.encoder = Encoder(n, self.device)
        self.decoder = Decoder(n)

        self.epoch = 0

    def forward(self, img):
        img = self.encoder(img)
        img = self.decoder(img)

        return img


def train(vae, optimizer, loss, data, save_as_path, device, epochs):
    for epoch in range(vae.epoch, vae.epoch + epochs):

        for imgs, _ in data:

            imgs = imgs.float().to(device)

            optimizer.zero_grad()

            y_pred = vae(imgs)

            loss = loss(y_pred, imgs)

            loss.backward()
            optimizer.step()

        vae.epoch += 1
        print(f'{epoch}-{loss}')

        if not epoch % 50:
            torch.save(vae, f'{save_as_path}{epoch}.vae_model')
