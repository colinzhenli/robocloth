import torch
import torch.nn as nn
from collections import OrderedDict
import drjit as dr
@dr.wrap_ad(source='drjit', target='torch')
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, num_hidden_layers, outermost_linear, nonlinearity):
        super().__init__()
        self.layers = []
        self.layers.append(nn.Linear(in_features, hidden_features))
        self.layers.append(self.get_nonlinearity(nonlinearity))

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.layers.append(self.get_nonlinearity(nonlinearity))

        if outermost_linear:
            self.layers.append(nn.Linear(hidden_features, out_features))
        else:
            self.layers.append(nn.Linear(hidden_features, out_features))
            self.layers.append(self.get_nonlinearity(nonlinearity))
        
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

    def get_nonlinearity(self, type):
        if type == 'relu':
            return nn.ReLU()
        elif type == 'tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError(f"Nonlinearity '{type}' not implemented.")

@dr.wrap_ad(source='drjit', target='torch')
class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='relu', in_features=2, mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)

    def forward(self, model_input):
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = OrderedDict()
        
        x = coords
        for i, layer in enumerate(self.net):
            x = layer(x)
            if isinstance(layer, nn.Linear):
                activations[f'layer_{i}'] = x.clone().detach()
        
        return {'model_in': coords, 'model_out': x, 'activations': activations}
