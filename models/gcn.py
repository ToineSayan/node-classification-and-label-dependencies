import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math



class GCN(nn.Module):

    def __init__(self, opt, adj):
        super(GCN, self).__init__()
        self.opt = opt
        self.adj = adj
        self.dimensions = [opt['num_features']] + opt['hidden_dims'] + [opt['num_classes']] 
        self.num_layers = len(self.dimensions) - 1
        self.dropout = self.opt['dropout']
        self.input_dropout = self.opt['input_dropout']

        self.layers = []
        for i in range(self.num_layers):
            opt_layer = {
                "in_size": self.dimensions[i],
                "out_size": self.dimensions[i+1]
            }
            self.layers.append(ConvolutionLayer(opt_layer, adj))
        self.layers = nn.ModuleList(self.layers) #adds parameters of the layers to the current class


    def reset_parameters(self):
        for i in range(self.num_layers):
            self.layers[i].reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.input_dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if not i+1 == self.num_layers:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x

    def __str__(self):
        s = "GCN (Graph Convolutional Network):\n"
        s += "----------------------------------\n"
        s += "Size of features: {}\n".format(self.dimensions[0]) 
        s += "Number of classes: {}\n".format(self.dimensions[-1]) 
        s += "Number of layers: {}\n".format(self.num_layers) 
        # for i in range(self.num_layers):
        #     s += self.layers[i].__str__()
        return(s)



class ConvolutionLayer(nn.Module):

    def __init__(self, opt, adj):
        super(ConvolutionLayer, self).__init__()
        self.opt = opt
        self.in_size = opt['in_size']
        self.out_size = opt['out_size']
        self.adj = adj
        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.bias = Parameter(torch.zeros(self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.out_size)
        # self.weight.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        #Forward pass in GCN is H^{i+1} = sigma(W^{i}*H^{i}*A + b)
        #This method does only the multiplication inside the activation function
        m = torch.mm(x, self.weight) # matrix mutliplication: x*weight
        m = torch.spmm(self.adj, m) # sparse matrix multiplication
        m.add_(self.bias)
        return m

