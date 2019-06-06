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

def main():
    np.random.seed(0)
    n_mix = 2
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    model = torch.load('moon_made.pt').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    parameters = list(model.parameters())

    model_parameters = filter(lambda p: p.requires_grad, parameters)
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total params:', params)

    n = 1000
    x, y = make_moons(n, noise=0.1)
    fisher_scores = np.zeros((n, params))

    for i in tqdm(range(len(x))):
        x_batch = torch.FloatTensor(x[[i]])
        optimizer.zero_grad()
        pi, mu, logstd = model(x_batch)
        dist = torch.distributions.Normal(mu, logstd.exp())
        log_prob = dist.log_prob(x_batch.unsqueeze(-1).repeat(1, 1, n_mix))
        log_prob = torch.log((log_prob.exp() * pi).sum(-1) + 1e-8)
        log_prob = log_prob.sum()
        log_prob.backward()

        fisher_score = torch.cat([p.grad.view(-1) for p in parameters])
        fisher_score = fisher_score.numpy()
        fisher_scores[i] = fisher_score
    print((fisher_scores > 1e-8).sum())
    # fisher_scores[fisher_scores < 1e-8] = 0

    idx = 0
    dot_products = np.dot(fisher_scores, fisher_scores[idx])
    nearest_pts = np.argsort(dot_products)[::-1]

    fig, axs = plt.subplots(3, 3)
    for i in range(9):
        pts = x[nearest_pts[:i*100]]
        axs[i // 3, i % 3].scatter(pts[:,0], pts[:,1], c='blue', alpha=0.5)
        axs[i // 3, i % 3].scatter(x[[idx],0], x[[idx],1], c='orange')

    plt.show()

if __name__ == '__main__':
    main()
