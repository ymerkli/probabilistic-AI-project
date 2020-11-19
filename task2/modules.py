import numpy as np
import torch
import os
import sys
import math
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from tqdm import trange, tqdm
from utils import *

JITTER = 1e-6

class Densenet(torch.nn.Module):
    '''
    Simple module implementing a feedforward neural network with
    num_layers layers of size width and input of size input_size.
    '''
    def __init__(self, input_size, num_layers, width):
        super().__init__()
        input_layer = torch.nn.Sequential(nn.Linear(input_size, width),
                                           nn.ReLU())
        hidden_layers = [nn.Sequential(nn.Linear(width, width),
                                    nn.ReLU()) for _ in range(num_layers)]
        output_layer = torch.nn.Linear(width, 10)
        layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out

    def predict_class_probs(self, x):
        probs = F.softmax(self.forward(x), dim=1)
        return probs


class BayesianLayer(torch.nn.Module):
    '''
    Module implementing a single Bayesian feedforward layer.
    The module performs Bayes-by-backprop, that is, mean-field
    variational inference. It keeps prior and posterior weights
    (and biases) and uses the reparameterization trick for sampling.
    '''
    def __init__(self, input_dim, output_dim, prior_mu=0, prior_sigma=0.1, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias

        # TODO: enter your code here
        self.prior_mu = nn.Parameter(torch.Tensor(1)).data.fill_(prior_mu)
        self.prior_sigma = prior_sigma
        self.prior_logsigma = nn.Parameter(torch.Tensor(1)).data.fill_(math.log(prior_sigma))

        self.weight_mu = nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        self.weight_logsigma = nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        self.register_buffer('weight_eps', None)

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.zeros(output_dim))
            self.bias_logsigma = nn.Parameter(torch.zeros(output_dim))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_logsigma', None)

        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        stdv = 0.1
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_logsigma.data.fill_(math.log(self.prior_sigma))
        if self.use_bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_logsigma.data.fill_(math.log(self.prior_sigma))

    def forward(self, inputs):
        weight = self.weight_mu + (torch.exp(self.weight_logsigma) + JITTER) * torch.randn_like(self.weight_logsigma)
        bias = None
        if self.use_bias:
            bias = self.bias_mu + (torch.exp(self.bias_logsigma) + JITTER) * torch.randn_like(self.bias_logsigma)

        return F.linear(inputs, weight, bias)

    def kl_divergence(self):
        '''
        Computes the KL divergence between the priors and posteriors for this layer.
        '''
        kl_loss = self._kl_divergence(self.weight_mu, self.weight_logsigma)
        if self.use_bias:
            kl_loss_bias = self._kl_divergence(self.bias_mu, self.bias_logsigma)
            kl_loss += kl_loss_bias

        return kl_loss

    def _kl_divergence(self, mu, logsigma):
        '''
        Computes the KL divergence between one Gaussian posterior
        and the Gaussian prior.
        '''
        # extract scalar values from torch tensor
        prior_logsigma = self.prior_logsigma.data[0]
        prior_mu = self.prior_mu.data[0]
        kl = logsigma - prior_logsigma + \
            (math.exp(prior_logsigma)**2 + (prior_mu - mu)**2) / (2*torch.exp(logsigma)**2) - 0.5

        return kl.mean()


class BayesNet(torch.nn.Module):
    '''
    Module implementing a Bayesian feedforward neural network using
    BayesianLayer objects.
    '''
    def __init__(self, input_size, num_layers, width, prior_mu=0, prior_sigma=0.1,):
        super().__init__()
        self.output_dim = 10

        if type(width) == list:
            input_layer = torch.nn.Sequential(BayesianLayer(input_size, width[0], prior_mu, prior_sigma),
                                           nn.ReLU())
            hidden_layers = []
            for i in range(len(width)-1):
                hidden_layers.append(
                    torch.nn.Sequential(BayesianLayer(width[i], width[i+1], prior_mu, prior_sigma))
                )
            output_layer = BayesianLayer(width[-1], self.output_dim, prior_mu, prior_sigma)
        else:
            input_layer = torch.nn.Sequential(BayesianLayer(input_size, width, prior_mu, prior_sigma),
                                           nn.ReLU())
            hidden_layers = [nn.Sequential(BayesianLayer(width, width, prior_mu, prior_sigma),
                                nn.ReLU()) for _ in range(num_layers)]
            output_layer = BayesianLayer(width, self.output_dim, prior_mu, prior_sigma)

        layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*layers)

    def save(self, file_path='bayesnet.pt'):
        print("Saving BayesNet model to ", file_path)
        torch.save(self.net, file_path)

    def forward(self, x):
        x = x.squeeze()
        return self.net(x)

    def predict_class_probs(self, x, num_forward_passes=10):
        x = x.squeeze()
        assert x.shape[1] == 28**2
        batch_size = x.shape[0]

        # TODO: make n random forward passes
        # compute the categorical softmax probabilities
        # marginalize the probabilities over the n forward passes
        probs = x.data.new(num_forward_passes, x.shape[0], self.output_dim)

        for i in range(num_forward_passes):
            y = self.forward(x)
            probs[i] = y

        # average over the num_forward_passes dimensions
        probs = probs.mean(dim=0, keepdim=False)

        assert probs.shape == (batch_size, 10)
        return F.softmax(probs, dim=1)

    def kl_loss(self, reduction='mean'):
        '''
        Computes the KL divergence loss for all layers.
        '''
        # TODO: enter your code here
        kl = torch.Tensor([0])
        kl_sum = torch.Tensor([0])
        n = torch.Tensor([0])

        for m in self.modules():
            if isinstance(m, (BayesianLayer)):
                kl = m.kl_divergence()
                kl_sum += kl
                n += len(m.weight_mu.view(-1))
                if m.use_bias:
                    n += len(m.bias_mu.view(-1))
        if reduction == 'mean':
            return kl_sum/n
        elif reduction == 'sum':
            return kl_sum
        else:
            raise ValueError("Error: {0} is not a valid reduction method".format(reduction))