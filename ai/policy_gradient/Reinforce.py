import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Reinforce(nn.Module):
    def __init__(self, layer_dims, device):
        #Initialize super class
        super(Reinforce, self).__init__()

        #Create layers
        self.layer_dims = layer_dims
        self.layers = nn.ModuleList(self.create_layers(layer_dims))

        #Set device
        self.device = device
        self.to(self.device)
        
        
    def create_layers(self, dimensions):
        layers = []
        
        for i in range(1, len(dimensions)):
            in_dim = dimensions[i-1]
            out_dim = dimensions[i]
            
            layers.append(nn.Linear(in_dim, out_dim))

        return layers
    
    def forward_layers(self, layers, hidden_activation, last_activation, input):
        #Feed state through each layer (except last layer) in network
        for layer in layers[:-1]:
            input = hidden_activation(layer(input))
            
        #Feed through last layer with last activation function
        output = last_activation(layers[-1](input))
        
        return output
        

    def forward(self, x):
        #If input is already a tensor, use it directly
        if type(x) is T.Tensor:
            input = x.to(T.float).to(self.device)
        #Else, make input a tensor
        else:
            #Turn state into tensor
            input = T.tensor(x).to(T.float).to(self.device)


        #Feed state through network
        action_probs = self.forward_layers(list(self.layers.children()),
                                           T.tanh, lambda x: F.softmax(x, dim=-1), 
                                           input)

        #Return action probalities and value of state
        return action_probs