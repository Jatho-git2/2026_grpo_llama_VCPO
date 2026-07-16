import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer
from reparameterization import LinearReparameterization
import peft

class BayesianLoraLinear(nn.Module, LoraLayer):
    def __init__(self, base_layer: nn.Module, adapter_name: str, r: int, lora_alpha: int, **kwargs):
        super().__init__()
        # 1. Initialize structural properties required by PEFT
        LoraLayer.__init__(self, base_layer)
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # 1.5 Store the base layer's name for reference in PEFT tracking
        #self.base_name = base_layer.name

        scaling = lora_alpha / r
        # with torch.no_grad():
        #     self.base_layer.weight.copy_(base_layer.weight - scaling*structured_prior_dict[(base_layer.name +".lora_B.default.weight")].mean@structured_prior_dict[(base_layer.name +".lora_A.default.weight")].mean)
        
        # 2. Setup internal PEFT tracking dicts
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        
        # # 3. Extract prior configurations
        # self.prior_params = {
        #     "prior_mean": kwargs.get("prior_mean", 0.0),
        #     "prior_variance": kwargs.get("prior_variance", 1.0),
        #     "posterior_mu_init": kwargs.get("posterior_mu_init", 0.0),
        #     "posterior_rho_init": kwargs.get("posterior_rho_init", -3.0),
        # }
        
        # 4. Instantiate the adapter layers
        self.update_layer(adapter_name, r, lora_alpha, **kwargs)

    def get_rho(var):
        """
        var is represented by the square of the softplus function  'sigma = log(1 + exp(rho))' to make sure it 
        remains always positive and non-transformed 'rho' gets updated during backprop.
        """
        rho = torch.log(torch.expm1(torch.sqrt(var)) + 1e-20)
        return rho

    def update_layer(self, adapter_name: str, r: int, lora_alpha: int, config: peft.LoraConfig, **kwargs):
        """Called automatically by PEFT to populate or modify adapter components."""
        if r <= 0:
            raise ValueError(f"r must be a positive integer, got {r}")
            
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        # Ensure that the LoRA dropout is applied only if specified
        lora_dropout = config.lora_dropout
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer  = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        self.scaling[adapter_name] = lora_alpha / r
        
        # Low-rank decomposition using Bayesian Reparameterization Layers
        # lora_A maps: in_features -> rank (r)

        self.lora_A[adapter_name] = LinearReparameterization(
            in_features=self.in_features,
            out_features=r,
            bias=False,
            prior_mean = structured_prior_dict[(self.base_layer.name +".lora_A.default.weight")].mean,
            prior_variance = structured_prior_dict[(self.base_layer.name +".lora_A.default.weight")].variance,
            posterior_mu_init = structured_prior_dict[(self.base_layer.name +".lora_A.default.weight")].mean,
            posterior_rho_init = get_rho(structured_prior_dict[(self.base_layer.name +".lora_A.default.weight")].variance)
        )

        # lora_B maps: rank (r) -> out_features
        self.lora_B[adapter_name] = LinearReparameterization(
            in_features=r,
            out_features=self.out_features,
            bias=False,
            prior_mean = structured_prior_dict[(self.base_layer.name +".lora_B.default.weight")].mean,
            prior_variance = structured_prior_dict[(self.base_layer.name +".lora_B.default.weight")].variance,
            posterior_mu_init = structured_prior_dict[(self.base_layer.name +".lora_B.default.weight")].mean,
            posterior_rho_init = get_rho(structured_prior_dict[(self.base_layer.name +".lora_B.default.weight")].variance)   
        )
        
        # Ensure base layer weights remain strictly frozen
        self.base_layer.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes deterministic frozen base path combined with stochastic LoRA path."""
        # Baseline deterministic forward pass
        result = self.base_layer(x)
        
        # Check active adapters
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            # Dropout is applied to the input before passing through the LoRA layers
            # Grab adapter weights and tracking values
            lora_A_layer = self.lora_A[active_adapter]
            lora_B_layer = self.lora_B[active_adapter]
            scaling = self.scaling[active_adapter]
            
            # Stochastic forward pass utilizing the reparameterization trick
            # Output represents one Monte Carlo sample from the weight distribution
            lora_A_out, _ = lora_A_layer(self.lora_dropout[active_adapter](x))
            lora_out, _ = lora_B_layer(lora_A_out)
            
            # Combine updates scaled by the low-rank multiplier
            result += lora_out * scaling
            
        return result