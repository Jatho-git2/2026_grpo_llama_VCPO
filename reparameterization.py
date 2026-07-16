# Copyright (C) 2024 Intel Labs
#
# BSD-3-Clause License
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Linear Reparameterization Layers with reparameterization estimator to perform
# variational inference in Bayesian neural networks. Reparameterization layers
# enables Monte Carlo approximation of the distribution over 'kernel' and 'bias'.
#
# Kullback-Leibler divergence between the surrogate posterior and prior is computed
# and returned along with the tensors of outputs after linear opertaion, which is
# required to compute Evidence Lower Bound (ELBO).
#
# @authors: Ranganath Krishnan
# ======================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from bayesian_torch.layers.base_variational_layer import BaseVariationalLayer_, get_kernel_size
import math

# Reparameterization trick for linear layers, specified for use in Bayesian LoRA layers.
class LinearReparameterization(BaseVariationalLayer_):
    def __init__(self,
                 in_features,
                 out_features,
                 prior_mean=0.0,
                 prior_variance=1.0,
                 posterior_mu_init=0.0,
                 posterior_rho_init=-3.0,
                 mc=True,
                 dtype = torch.float16
                 ):
        """
        Implements Linear layer with reparameterization trick.

        Inherits from bayesian_torch.layers.BaseVariationalLayer_

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_mean: float -> mean of the prior arbitrary distribution to be used on the complexity cost,
            prior_variance: float -> variance of the prior arbitrary distribution to be used on the complexity cost,
            posterior_mu_init: float -> init trainable mu parameter representing mean of the approximate posterior,
            posterior_rho_init: float -> init trainable rho parameter representing the sigma of the approximate posterior through softplus function,
            mc: bool -> whether to use Monte Carlo sampling for the forward pass (default: True)
        """
        super(LinearReparameterization, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init  # mean of weight
        # variance of weight --> sigma = log (1 + exp(rho))
        self.posterior_rho_init = posterior_rho_init
        self.mc = mc
        self.dtype=dtype

        self.mu_weight = Parameter(torch.zeros(out_features, in_features, dtype=self.dtype))
        self.rho_weight = Parameter(torch.zeros(out_features, in_features, dtype=self.dtype))
        self.register_buffer('prior_weight_mu',
                             torch.zeros(out_features, in_features, dtype=self.dtype),
                             persistent=False)
        self.register_buffer('prior_weight_sigma',
                             torch.zeros(out_features, in_features, dtype=self.dtype),
                             persistent=False)

        self.init_parameters()
    
    # cast all below at dtype torch.bfloat16
    def init_parameters(self):
        with torch.no_grad():
            self.prior_weight_mu.fill_(self.prior_mean)
            self.prior_weight_sigma.fill_(math.sqrt(self.prior_variance))

            self.mu_weight.fill_(self.posterior_mu_init)
            self.rho_weight.fill_(self.posterior_rho_init)

    def kl_loss(self):
        sigma_weight = torch.log1p(torch.exp(self.rho_weight))
        kl = self.kl_div(
            self.mu_weight,
            sigma_weight,
            self.prior_weight_mu,
            self.prior_weight_sigma)
        
        return kl
    
    def set_mc(self, mc=True):
        self.mc = mc
        
    def forward(self, input):
        
        out = F.linear(input, self.mu_weight, bias=None)
        if self.mc:
            sigma_weight = torch.log1p(torch.exp(self.rho_weight))
            eps = torch.randn_like(out)
            delta_out = torch.sqrt(F.linear((input)**2, (sigma_weight)**2, bias=None))*eps
            out += delta_out            

        return out
    
    def get_var(self):
        return torch.square(torch.log1p(torch.exp(self.rho_weight)))
    
    def get_mean(self):
        return self.mu_weight
    