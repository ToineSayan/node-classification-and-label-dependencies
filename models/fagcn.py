
from numpy import dtype
import torch
from torch import nn
import torch.nn.functional as F
from datetime import datetime




class FAGCN(nn.Module):

    def __init__(self, opt, adj):
        super(FAGCN, self).__init__()
        self.opt = opt
        self.adj = adj

        self.dimensions = [opt['num_features']] + opt['hidden_dims'] + [opt['num_classes']]
        if not len(self.dimensions) == 3:
            raise NameError('Wrong dimensions for FAGCN: 2 layers network, dimensions must be [in_dim, hidden_dim, out_dim]')
        self.num_layers = 2 # Fixed, we use layer to refer to projections followed by a non-linear activation
        self.dropout = self.opt['dropout']
        self.input_dropout = self.opt['input_dropout']
        self.num_prop = self.opt['fagcn_num_prop']
        self.epsilon = self.opt['fagcn_epsilon']

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.dimensions[0], self.dimensions[1]))
        self.layers.append(nn.Linear(self.dimensions[1], self.dimensions[2]))
        
        self.propagations = nn.ModuleList()
        for i in range(self.num_prop):
            opt_prop = {
                'hidden_dim': self.dimensions[1],
                'dropout': self.dropout
            }
            self.propagations.append(FAGCNPropagation(opt_prop, self.adj))
        
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers):
            # nn.init.xavier_normal_(self.layers[i].weight, gain=1.414)
            nn.init.xavier_uniform_(self.layers[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.layers[i].bias)
        for i in range(self.num_prop):
            self.propagations[i].reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.input_dropout, training=self.training)
        x = F.relu(self.layers[0](x))
        x = F.dropout(x, self.dropout, training=self.training)
        x0 = x
        for i in range(self.num_prop):
            x = self.propagations[i](x)
            x = self.epsilon*x0 + x
        x = self.layers[1](x)
        return x

    def output_alpha_coefs(self, y_target):
        in_labels = y_target[self.adj._indices()[0]]
        out_labels = y_target[self.adj._indices()[1]]
        alpha_vals = self.propagations[-1].alpha_vals # we get the alpha coefs of the last layer
        alpha_vals = alpha_vals.detach()
        alpha_equ = alpha_vals[torch.where(in_labels == out_labels)]
        alpha_neq = alpha_vals[torch.where(in_labels != out_labels)]
        time_str = str(datetime.now())
        f = open('fagcn_alpha_' + time_str + '.out', 'w')
        str_out = '['
        for val in alpha_equ.cpu().numpy():
            str_out += str(val) + ','
        str_out = str_out[:-1] + ']\n['
        for val in alpha_neq.cpu().numpy():
            str_out += str(val) + ','
        str_out = str_out[:-1] + ']'
        f.write(str_out)
        f.close()
        return None

    
    def adj_init(self, adj, add_self_loops = False):

        self.adj = adj

        # # Add self-loops 
        # indices = self.adj._indices()
        # idx = indices[0] == indices[1] # Calculate a mask to keep all indices except the ones on the diagonal
        # values = self.adj._values()
        # values[idx] = 1.0
        # indices = torch.vstack((indices[0], indices[1]))
        # # values = self.adj._values()[idx]
        # self.adj = torch.sparse.FloatTensor(indices, values, self.adj.shape)



        # # Remove self links (calculates A~)
        # indices = self.adj._indices()
        # idx = indices[0] != indices[1] # Calculate a mask to keep all indices except the ones on the diagonal
        # indices = torch.vstack((indices[0][idx], indices[1][idx]))
        # values = self.adj._values()[idx]
        # self.adj = torch.sparse.FloatTensor(indices, values, self.adj.shape)

        # # Calculate the square root inverse diagonal matrix of degrees (D^(-1/2))
        # degrees = torch.sparse.sum(self.adj, dim=1).to_dense().clamp(min=1)
        # inv_degrees = torch.pow(degrees, -0.5)
        # ind = torch.nonzero(inv_degrees).t()
        # indices = torch.vstack((ind, ind))
        # D = torch.sparse.FloatTensor(indices, inv_degrees, self.adj.shape) 
        

        # # Calculate Normalized adjacency matrix (D^(-1/2)A~D^(-1/2))
        # self.adj = torch.sparse.mm(self.adj, D)
        # self.adj = torch.sparse.mm(D, self.adj)

        return None



class FAGCNPropagation(nn.Module):

    def __init__(self, opt, adj):
        super(FAGCNPropagation, self).__init__()
        self.opt = opt
        self.hidden_dim = opt['hidden_dim']
        self.dropout = opt['dropout']
        self.adj = adj
        self.l = len(self.adj._indices()[0])
        self.gate1 = nn.Linear(self.hidden_dim, 1, bias=False) # bias = false as in pytorch geometric implementation
        self.gate2 = nn.Linear(self.hidden_dim, 1, bias=False) # bias = false as in pytorch geometric implementation
        self.alpha_vals = None
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_normal_(self.gate1.weight, gain=1.414)
        # nn.init.xavier_normal_(self.gate2.weight, gain=1.414)

        nn.init.xavier_uniform_(self.gate1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.gate2.weight, gain=nn.init.calculate_gain('relu'))


        



    def propagate(self, x):
        in_indices = self.adj._indices()[0]
        out_indices = self.adj._indices()[1]

        x1 = self.gate1(x)
        x2 = self.gate2(x)
        m = torch.tanh(torch.add(x1[in_indices],x2[out_indices])).squeeze()

        # m = torch.cat((x[in_indices], x[out_indices]), dim=1)
        # m = torch.tanh(self.gate(m)).squeeze()

        self.alpha_vals = m

        m = m * self.adj._values()
        m = F.dropout(m, self.dropout, training=self.training)

        adj_alpha = torch.sparse.FloatTensor(self.adj._indices(), m, self.adj.shape)
        result = torch.sparse.mm(adj_alpha, x)

        return result


    def forward(self, x):
        #Forward pass in FAGCN is 
        m = self.propagate(x)
        return m