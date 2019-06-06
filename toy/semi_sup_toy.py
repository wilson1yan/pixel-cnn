from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__(in_features, out_features, bias=bias)
        self.mask = None

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    def set_mask(self, mask):
        assert self.weight.size() == mask.size()
        self.mask = mask

class MADE(nn.Module):
    # n = number of rvs, d = dimension of random variable
    def __init__(self, n, d, hidden_sizes):
        super(MADE, self).__init__()
        self.linears = nn.ModuleList()
        output_dim = n * d

        self.hidden_sizes = hidden_sizes
        self.layer_sizes = hidden_sizes + [output_dim]

        prev_h = n
        for h in self.layer_sizes:
            self.linears.append(MaskLinear(prev_h, h))
            prev_h = h

        self.L = len(hidden_sizes) + 1
        self.n = n
        self.d = d

        self.create_masks()

        for mask, linear in zip(self.masks, self.linears):
            linear.set_mask(torch.FloatTensor(mask.astype('float32')))

        self.ordering = self.m[0]

    def create_masks(self):
        self.m = []
        self.m.append(np.random.permutation(np.arange(self.n)))
        for l, h in zip(range(1, self.L), self.hidden_sizes):
            min_k = np.min(self.m[l - 1])
            self.m.append(np.random.choice(np.arange(min_k, self.n-1), size=h))
        self.m.append(self.m[0])

        self.masks = [self.m[l][:, np.newaxis] >= self.m[l-1][np.newaxis, :]
                     for l in range(1, self.L)]
        self.masks.append(self.m[self.L][:, np.newaxis] > self.m[self.L-1][np.newaxis, :])
        self.masks[-1] = np.repeat(self.masks[-1], self.d, axis=0)

    def forward(self, x):
        for linear in self.linears[:-1]:
            x = F.relu(linear(x))
        x = self.linears[-1](x)
        x = x.view(x.size(0), self.n, -1)
        pi, mu, logstd = x.chunk(3, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return pi, mu, logstd

    def sample(self, n, device):
        samples = torch.zeros((n, self.n))
        for i in range(self.n):
            pi, mu, logstd = self(samples)
            pi, mu, logstd = pi[:, i], mu[:, i], logstd[:, i]
            pi_idx = torch.multinomial(pi, 1)
            mu, logstd = torch.gather(mu, 1, pi_idx), torch.gather(logstd, 1, pi_idx)
            mu, logstd = mu.squeeze(-1), logstd.squeeze(-1)
            dist = torch.distributions.Normal(mu, logstd.exp())
            samples[:, i] = dist.sample()
        return samples

def plot_labeled(x, y):
    cdict = {0: 'blue', 1: 'orange'}

    plt.figure()
    for g in np.unique(y):
        ix = np.where(y == g)
        plt.scatter(x[ix,0], x[ix,1], c=cdict[g], label=g)
    plt.legend()
    plt.show()

def plot_unlabeled(x):
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    n_mix = 2
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    model = MADE(2, 3 * n_mix, [256, 256, 256]).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=5e-4)

    x, y = make_moons(10000, noise=0.1)

    for e in range(args.epochs):
        idx = np.random.permutation(len(x))
        x, y = x[idx], y[idx]
        train_losses = []
        pbar = tqdm(total=len(x))
        for i in range(0, len(x), args.batch_size):
            x_batch = torch.FloatTensor(x[i:i+args.batch_size]).to(device)
            pi, mu, logstd = model(x_batch)
            dist = torch.distributions.Normal(mu, logstd.exp())
            log_prob = dist.log_prob(x_batch.unsqueeze(-1).repeat(1, 1, n_mix))
            log_prob = torch.log((log_prob.exp() * pi).sum(-1) + 1e-8)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            pbar.update(x_batch.size(0))
            pbar.set_description('Epoch {}, loss {:.4f}'.format(e, np.mean(train_losses)))
        pbar.close()

    torch.save(model, 'moon_made.pt')

    with torch.no_grad():
        samples = model.sample(1000, device).cpu().numpy()
        plot_unlabeled(samples)

if __name__ == '__main__':
    main()
