def get_default_config():
    default_config = dict()

    default_config['dataset'] = {'type':str, 'default':'CORA'}
    default_config['normalization'] = {'type':str, 'default':'global-L1', 'help':'Normalization of features. "global-L1" for L1 norm, "global-L2" for L2 norm, "None" for no normalization'}
    default_config['binarize'] = {'type':bool, 'default':True, 'help':'Whether to binarize the features'}
    # General args
    default_config['seed'] = {'type':int, 'default':42}
    # Args for graph 
    default_config['normalization_trick'] = {'type': str, 'default':'None'}
    default_config['to_symmetric'] = {'type': bool, 'default':True, 'help': 'Wether to transform the adjacency matrix into a symmetric one'}
    default_config['add_self_links'] = {'type': bool, 'default':True, 'help': 'Wether to add self links in the adjacency matrix'}
    # Args for the architecture to use (model base (q)}
    default_config['model_base_q'] = {'type':str, 'default':'MLP', 'help': 'Architecture to use to train the model'}
    default_config['hidden_dims_base_q'] = {'type':int, 'nargs':'*', 'default':"8", 'help':'Hidden dimensions. If no argument or "", no hidden dimension (1 layer)'}
    default_config['input_dropout_base_q'] = {'type':float, 'default':0.2, 'help':'Input dropout rate.'}
    default_config['dropout_base_q'] = {'type':float, 'default':0.8, 'help':'Dropout rate.'}
    default_config['gs_aggregator'] = {'type':str, 'default':'mean', 'help': 'Aggregator for GraphSAGE architecture'}
    default_config['gs_sample_size'] = {'type':int, 'default':5, 'help': 'Number of neighbors to sample for GraphSAGE architecture'}
    default_config['fagcn_epsilon'] = {'type':float, 'default':0.5, 'help': 'Epsilon value for FAGCN architecture'}
    default_config['fagcn_num_prop'] = {'type':int, 'default':2, 'help': 'Number of propagations for FAGCN architecture'}
    # Args for the trainer (model base (q)}
    default_config['optimizer_base_q'] = {'type':str, 'default':'adam', 'help':'Optimizer'}
    default_config['lr_base_q'] = {'type':float, 'default':0.01, 'help':'Learning rate'}
    default_config['decay_base_q'] = {'type':float, 'default':5e-4, 'help':'Weight decay (L2-regularization)'}
    default_config['decay_policy'] = {'type':int, 'nargs':'*', 'default':[-1], 'help':'List of layers on which the L2-regularization should be applied. Default value [-1], the regularization is applied on all layers. The first layer of a model is layer 0 and the last is layer L-1 if L is the total number of layers.'}

    default_config['GMNN_architecture'] = {'type':bool, 'default':False, 'help':'Activate the EM loops (GMNN architecture)'}
    default_config['EM_loops'] = {'type':int, 'default':3, 'help':'Number of EM loops (1 loop = a traing of q_theta + a training of p_phi)'}

    # Args for the architecture to use (model p}
    default_config['model_p'] = {'type':str, 'default':'MLP', 'help': 'Architecture to use to train the model'}
    default_config['hidden_dims_p'] = {'type':int, 'nargs':'*', 'default':"8", 'help':'Hidden dimensions for GNNp. If no argument or "", no hidden dimension (1 layer)'}
    default_config['input_dropout_p'] = {'type':float, 'default':1, 'help':'Input dropout rate.'}
    default_config['dropout_p'] = {'type':float, 'default':1, 'help':'Dropout rate.'}
    default_config['draw'] = {'type':str, 'default':'exp', 'help':'Method for drawing object labels, max for max-pooling, smp for sampling.'}
    default_config['use_features_p'] = {'type':bool, 'default': True, 'help':'Wether to use input features to train p or only predictions from GNNq'}
    default_config['use_gold_p'] = {'type':bool, 'default':False, 'help':'Whether to use gold values for training instances when training GNNp'}
    default_config['fagcn_epsilon_p'] = {'type':float, 'default':0.2, 'help': 'Epsilon value for FAGCN architecture'}
    # Args for the trainer (model p}
    default_config['optimizer_p'] = {'type':str, 'default':'adam', 'help':'Optimizer'}
    default_config['lr_p'] = {'type':float, 'default':0.01, 'help':'Learning rate'}
    default_config['decay_p'] = {'type':float, 'default':5e-4, 'help':'Weight decay (L2-regularization)'}
    default_config['tau_p'] = {'type':float, 'default':1.0, 'help':'Temperature for GNNp'}

    # Args for the splitting strategy to use to create train, validation, and test sets
    default_config['splitting_method'] = {'type':str, 'default':'random-fixed', 'help':'Splitting strategy to create train, val., and test sets'}
    default_config['split_name'] = {'type':str, 'default':'fixed', 'help':'Name of the split to use'}
    default_config['train_set_size'] = {'type':int, 'default':140, 'help':'Number of observations to include in the train set (ignored if \'random-fixed\' method of splitting)'}
    default_config['val_set_size'] = {'type':int, 'default':500, 'help':'Number of observations to include in the validation set'}
    default_config['test_set_size'] = {'type':int, 'default':1000, 'help':'Number of observations to include in the test set'}
    default_config['num_nodes_per_class'] = {'type':int, 'default':20, 'help':'Number of observations (used only for \'random-fixed\' method of splitting) '}

    # Args for training
    default_config['epochs'] = {'type':int, 'default':100000, 'help':'Number of training epochs'}
    default_config['early_stopping'] = {'type':int, 'default':50, 'help':'Patience (used in early stopping criterion)'}
    default_config['iterations'] = {'type':int, 'default':1, 'help':'Number of trainings'}
    default_config['EM_epochs'] = {'type':int, 'default':100, 'help':'Number of training epochs for EM loops'}
    default_config['saved_model_q'] = {'type':str, 'default': None, 'help':'Wether to load or save a saved model. Values "load", "save", None. Default None'}
    default_config['path_saved_model_q'] = {'type':str, 'default': None, 'help':'path to load or save a model q'}
    default_config['save_models_during_EM_phases'] = {'type':bool, 'default':True, 'help':'During EM phases, wether to save or not GNNq and GNNp models and optimisers when the validation accuracy reaches a maximum'}

    return default_config