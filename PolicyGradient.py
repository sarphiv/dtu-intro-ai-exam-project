import torch as T
import torch.nn as nn
import torch.nn.functional as F

class PolicyGradient(nn.Module):
    def __init__(self, layer_dims, device):
        #Initialize super class
        super(PolicyGradient, self).__init__()

        #Create layers
        self.layer_dims = layer_dims

        for i in range(1, len(layer_dims)):
            in_dim = layer_dims[i-1]
            out_dim = layer_dims[i]
            
            setattr(self, f"layer{i}", nn.Linear(in_dim, out_dim))

        #Set device
        self.device = device
        self.to(self.device)


    def forward(self, x):
        #If input is already a tensor, use it directly
        if type(x) is T.Tensor:
            input = x.to(T.float).to(self.device)
        #Else, make input a tensor
        else:
            #Turn state into tensor
            input = T.tensor(x).to(T.float).to(self.device)


        #Feed state through each layer (except last layer) in network
        layers = list(self.children())
        for layer in layers[:-1]:
            input = F.relu(layer(input))

        #Feed through last layer with softmax
        output = F.softmax(layers[-1](input), dim=-1)


        #Return output (logits) of network
        return output
