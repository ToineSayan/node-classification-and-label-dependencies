from torch import nn
import torch.nn.functional as F



# define NN architecture
class MLP(nn.Module):
    def __init__(self, opt, adj = None):
        super(MLP,self).__init__()
        self.opt = opt
        self.in_size = self.opt['num_features'] 
        self.out_size = self.opt['num_classes'] 
        self.dimensions = [opt['num_features']] + opt['hidden_dims'] + [opt['num_classes']]
        self.num_layers = len(self.dimensions) - 1
        self.dropout = self.opt['dropout']
        self.input_dropout = self.opt['input_dropout']
        layers = []
        # Adding the layers
        for i in range(self.num_layers):
            in_s = self.dimensions[i]
            out_s = self.dimensions[i+1]
            # linear layer (in_s -> out_s)
            layers.append(nn.Linear(in_s, out_s)) # bias is true for nn.Linear
        self.layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for i in range(self.num_layers):
            # self.layers[i].reset_parameters()
            nn.init.xavier_uniform_(self.layers[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.layers[i].bias)
        
    def forward(self,x):
        for i in range(self.num_layers):
            x = F.dropout(x, self.input_dropout, training=self.training)
            x = self.layers[i](x)
            if not i+1 == self.num_layers:
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(x)
        return x


