# List of all possible parameters in configuration files and their default values:

All these parameters are defined in 

- 'dataset': {'type':str, 'help': 'dataset among "CORA-ORIG", "CITESEER-ORIG", "PUBMED-ORIG" and "WIKIVITALS_NEW"'}
- 'normalization': {'type':str, 'help':'Normalization of features. "global-L1" for L1 norm, "global-L2" for L2 norm, "None" for no normalization'}
- 'binarize': {'type':bool, 'help':'Whether to binarize or not the features'}
## General args
- 'seed': {'type':int}
## Args for graph 
- 'normalization_trick': {'type': str, 'help': 'Normalization trick to use among "normalize_D-1", "normalize_D-0.5" or None'}
- 'to_symmetric': {'type': bool, 'default':True, 'help': 'Wether to transform or not the adjacency matrix into a symmetric one'}
- 'add_self_links': {'type': bool, 'default':True, 'help': 'Wether to add self links or not in the adjacency matrix'}

## Args for the architecture to use (GNN_theta}
- 'model_base_q': {'type':str, 'help': 'Architecture to use to train the model among "MLP", "GCN" or "FAGCN"'}
- 'hidden_dims_base_q': {'type':int, 'nargs':'*', 'help':'Hidden dimensions. If no argument or "", no hidden dimension (1 layer). Use "[16,8]" for 2 hidden dimensions 16 and 8 (3-layer model)'}
- 'input_dropout_base_q': {'type':float, 'help':'Input dropout rate.'}
- 'dropout_base_q': {'type':float, 'help':'Dropout rate.'}
- 'fagcn_epsilon': {'type':float, 'help': 'Epsilon value for FAGCN architecture'}
- 'fagcn_num_prop': {'type':int, 'help': 'Number of propagations for FAGCN architecture'}

## Args for the trainer (GNN_theta}
- 'optimizer_base_q': {'type':str, 'help':'Optimizer'}
- 'lr_base_q': {'type':float, 'help':'Learning rate'}
- 'decay_base_q': {'type':float, 'help':'Weight decay (L2-regularization)'}
- 'decay_policy': {'type':int, 'nargs':'*', 'help':'List of layers on which the L2-regularization should be applied. Default value [-1], the regularization is applied on all layers. The first layer of a model is layer 0 and the last is layer L-1 if L is the total number of layers.'}

## Args to activate GMNN
- 'GMNN_architecture': {'type':bool, 'help':'Activate or not the EM loops (GMNN architecture)'}
- 'EM_loops': {'type':int, 'help':'Number of EM loops (1 loop = a traing of GNN_theta + a training of GNN_phi)'}

## Args for the architecture to use (GNN_phi}
- 'model_p': {'type':str, 'help': 'Architecture to use to train the model among "MLP", "GCN" or "FAGCN"'}
- 'hidden_dims_p': {'type':int, 'nargs':'*', 'help':'Hidden dimensions for GNNp. If no argument or "", no hidden dimension (1 layer). Use "[16,8]" for 2 hidden dimensions 16 and 8 (3-layer model)'}
- 'input_dropout_p': {'type':float, 'help':'Input dropout rate.'}
- 'dropout_p': {'type':float, 'help':'Dropout rate.'}
- 'draw': {'type':str, 'help':'Method for drawing object labels, max for max-pooling, smp for sampling, exp for exp.'}
- 'use_features_p': {'type':bool, 'help':'Wether to use input features to train p or only predictions from GNNq'}
- 'use_gold_p': {'type':bool, 'help':'Whether to use gold values for training instances when training GNNp'}
- 'fagcn_epsilon_p': {'type':float, 'help': 'Epsilon value for FAGCN architecture'}

## Args for the trainer (GNN_phi}
- 'optimizer_p': {'type':str, 'help':'Optimizer'}
- 'lr_p': {'type':float, 'help':'Learning rate'}
- 'decay_p': {'type':float, 'help':'Weight decay (L2-regularization)'}
- 'tau_p': {'type':float, 'help':'Temperature for GNN_phi'}

## Args for the splitting strategy to use to create train, validation, and test sets
- 'splitting_method': {'type':str, 'help':'Splitting strategy to create train, val., and test sets'}
- 'split_name': {'type':str, 'help':'Name of the split to use'}
- 'train_set_size': {'type':int, 'help':'Number of observations to include in the train set (ignored if \'random-fixed\' method of splitting)'}
- 'val_set_size': {'type':int, 'help':'Number of observations to include in the validation set'}
- 'test_set_size': {'type':int, 'help':'Number of observations to include in the test set'}
- 'num_nodes_per_class': {'type':int, 'help':'Number of observations per class (used only for \'random-fixed\' method of splitting) '}

## Args for training
- 'epochs': {'type':int, 'help':'Number of training epochs'}
- 'early_stopping': {'type':int, 'help':'Patience (used in early stopping criterion)'}
- 'iterations': {'type':int, 'help':'Number of trainings'}
- 'EM_epochs': {'type':int, 'help':'Number of training epochs for EM loops'}
- 'saved_model_q': {'type':str, 'help':'Wether to load or save a saved model. Values "load", "save", None. Default None'}
- 'path_saved_model_q': {'type':str, 'help':'path to load or save a model q'}
- 'save_models_during_EM_phases': {'type':bool, 'help':'During EM phases, wether to save or not GNNq and GNNp models and optimisers when the validation accuracy reaches a maximum'}

