import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn


class VolumeTexture(nn.Module):
    '''
    map a 3d position to its BRDF parameters
    '''
    def __init__(
        self,
        num_levels=16,
        level_dim=4,
    ):
        super().__init__()
        per_level_scale = np.exp2(np.log2(1024 / 16) / (num_levels - 1))
        self.encoding = tcnn.Encoding(
            n_input_dims=3, 
            encoding_config={
                'otype': 'HashGrid', 
                'n_levels': num_levels, 
                'n_features_per_level': level_dim, 
                'log2_hashmap_size': 14, 
                'base_resolution': 16, 
                'per_level_scale': per_level_scale, 
            }
        )
        
        n_output_dims = 3
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims + 3,
            n_output_dims=n_output_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
    
    def forward(self, x):
        h = self.encoding(x).to(x)
        h = torch.cat([2 * x - 1, h], dim=-1)
        out = self.network(h).to(h)
        return out.abs()
