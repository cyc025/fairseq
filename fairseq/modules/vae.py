# Copyright (C) 2020  Ernie Chang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch as torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np




def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if args.cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


class VAE(nn.Module):
    def __init__(self,hidden_size):
        super(VAE, self).__init__()
        vae_dim = 20
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, 400)
        self.fc21 = nn.Linear(400, vae_dim)
        self.fc22 = nn.Linear(400, vae_dim)
        self.fc3 = nn.Linear(vae_dim, 400)
        self.fc4 = nn.Linear(400, hidden_size)

        self.proj1 = nn.Linear(50*512, 1)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def collect_z(self, hidden, is_plot=False):
        if is_plot:
            # Collect data points for scatter plot
            import numpy as np
            convert = lambda x: x.data.cpu().unsqueeze(0).view(1,-1).numpy()
            # Load
            latentVecs = np.load('plot_data/back_latent.vec.npy')
            # Update list
            if len(latentVecs)==0:
                latentVecs = convert(hidden)
            else:
                latentVecs = np.concatenate((latentVecs, convert(hidden)), axis=0)
                logger.info(latentVecs.shape)
            # save
            np.save('plot_data/back_latent.vec', latentVecs)

    def forward(self, x):
        import math
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # self.collect_z(z) # collect latent var
        new_x = self.decode(z)

        # make mask distribution
        m1 = nn.Sigmoid()
        m2 = nn.Softmax()

        ### DyMask start-end positions method
        mask_distribution = torch.squeeze(torch.max(x.view(x.size()[1],-1), 1, True),-1)
        # mask_distribution = torch.abs(m2(mask_distribution) * 100. - 1.)

        mask_distribution -= mask_distribution.min(0, keepdim=True)[0]
        mask_distribution /= mask_distribution.max(0, keepdim=True)[0]

        ### DyMask predict all mask positions method
        # mask_distribution = torch.round(m1(torch.mean(x,dim=2)))==0.

        # from fairseq import pdb; pdb.set_trace()

        return new_x, mu, logvar, mask_distribution
