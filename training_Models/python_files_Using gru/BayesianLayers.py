import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_var=0.02):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))

        self.weight_mu.requires_grad = True
        self.weight_rho.requires_grad = True
        self.bias_mu.requires_grad = True
        self.bias_rho.requires_grad = True

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -10)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -10)

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        epsilon_weight = torch.randn_like(weight_sigma)
        epsilon_bias = torch.randn_like(bias_sigma)
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        weight_var = weight_sigma ** 2
        bias_var = bias_sigma ** 2
        weight_kl = 0.5 * (weight_var + self.weight_mu ** 2 - 1 - weight_sigma)
        bias_kl = 0.5 * (bias_var + self.bias_mu ** 2 - 1 - bias_sigma)
        return weight_kl.sum() + bias_kl.sum()