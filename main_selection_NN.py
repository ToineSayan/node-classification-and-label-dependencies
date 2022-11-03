import sys, copy
import ast
import argparse
import torch
import json
import random
import load_datasets
from split import train_val_test_split
from trainer import Trainer
import early_stopping

from grid_search import GA_grid_search
from models.gcn import GCN
from models.perceptron import MLP 
from models.graphSAGE import GraphSAGE
from models.fagcn import FAGCN

from statistics import mean, stdev


# ----- GET ARGS -----
# The only argument is the name of the configuration file to use where all
# the parameters of the training to be executed are defined.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CORA-ORIG')
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--split_suffix', type=str, default='s')
args = parser.parse_args()

dataset_ = vars(args)['dataset']
model_ = vars(args)['model']
split_suffix_ = vars(args)['split_suffix']


# ----- CHECK IF CUDA AVAILABLE -----
torch.cuda.empty_cache()
cuda = torch.cuda.is_available()

# ----- LOAD CONFIGURATIONS -----
opt = json.load(open(f'./configs/grid_search/grid_search_base_{model_}.json', 'r', encoding='utf8'))
opt['dataset'] = dataset_
opt['model'] = model_

# ----- INITIATE GRID SEARCH CLASS -----
gs = GA_grid_search(opt["grid"])

# ----- LOAD DATASET -----
# Loading and pre-processing of data
X, y_target, adj = load_datasets.load_dataset(opt['dataset'])
num_nodes, num_features = X.shape
num_classes = torch.unique(y_target).shape[0]
opt['num_nodes'], opt['num_features'], opt['num_classes'] = num_nodes, num_features, num_classes
# Feature binarization (X unchanged if opt['binarize'] == False)
X = load_datasets.binarize_features(X, opt['binarize'])
# Feature normalization (X unchanged if opt['normalization'] == "None")
X = load_datasets.normalize_features(X, opt['normalization'])
# Binarize y_target 
y_target_bin = torch.nn.functional.one_hot(y_target)
# adjacency matrix normalization

adj_norm = load_datasets.transform_adjacency(
    adj, 
    opt['normalization_trick'],
    opt['to_symmetric'],
    opt['add_self_links']
    )



# ----- GENERATE THE LIST OF ALL INDICES -----
# Get the list of all indices
idx_all = torch.LongTensor([i for i in range(opt['num_nodes'])])

# ----- USE GPUs IF AVAILABLE -----
if cuda:
    X = X.cuda()
    y_target = y_target.cuda()
    y_target_bin = y_target_bin.cuda()
    adj_norm = adj_norm.cuda()
    idx_all = idx_all.cuda()

# ----- INITIALIZE THE LIST OF BEST INDIVIDUALS PER SPLIT -----
best_individuals = []

# Let's start to search for best configuration for each split
for split_idx in range(opt["k_num_splits"]): # for all pre-computed split...
    gs.reset() # reset grid search

    opt['split_name'] = f'split_{split_idx}_{split_suffix_}' # update split name (reconstruct it)

    # Get the list of indices for train, validation, and test sets
    idx_train, idx_val, idx_test = train_val_test_split(
        splitting_method = opt['splitting_method'],
        dataset = opt['dataset'].lower(),
        split_name = opt['split_name']
    )
    # Update train, validation, and test set sizes with their real values
    opt['train_set_size'] = idx_train.shape[0]
    opt['val_set_size'] = idx_val.shape[0]
    opt['test_set_size'] = idx_test.shape[0]


    # ----- USE GPUs IF AVAILABLE -----
    if cuda:
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Now that everything is defined, let's redefine the fitness (aka the evaluation 
    # of an individual) which is a function of the various elements previously calculated
    # and which varies from one split to another
    def fitness_GNN_individual(individual, grid_keys):

        # Initialization using the seed provided
        random.seed(opt['seed'])
        torch.manual_seed(opt['seed'])
        if cuda:
            torch.cuda.manual_seed(opt['seed'])

        current_config = opt
        unstring_list = lambda x: ast.literal_eval(x[4:])
        tmp = tuple([unstring_list(e) if (type(e) == str and e.startswith('LIST')) else e for e in individual])
        individual_config = dict(zip(grid_keys, tmp))

        for k,v in individual_config.items():
            current_config[k] = v
        
        # ----- INITIALIZE THE MODEL, THE TRAINER, AND SELECT AN EARLY STOPPING STRATEGY -----
        GNN_class  = getattr(sys.modules[__name__], opt['model'])
        GNN = GNN_class(opt, adj_norm) # adjacency matrix is ignored for MLP architecture 
        GNN_trainer = Trainer(opt, GNN)
        if cuda:
            GNN.cuda()
        # use the appropriate early stopping criterion
        Early_stopping_class = getattr(early_stopping, opt['early_stopping_type'])
        Early_stopping_criterion = Early_stopping_class( patience = opt['early_stopping'])


        # ----- TRAIN THE MODEL -----                
        results = []
        max_val_acc = 0.0 # Current maximum validation accuracy
        epoch = 0 # Current epoch
        stop_early = False # Boolean used to decide if early stopping
        # Note: it is not necessary to reset
        GNN_trainer.reset() # model is reset in trainer
        Early_stopping_criterion.reset()

        while not (stop_early or epoch >= opt['epochs']): # 2 stopping criterions: early stopping or max number of epoches reached
            loss = GNN_trainer.update(X, y_target, idx_train)
            train_loss, correct_train, preds_train, accuracy_train = GNN_trainer.evaluate(X, y_target, idx_train)
            val_loss, correct_val, preds_val, accuracy_val = GNN_trainer.evaluate(X, y_target, idx_val)
            test_loss , correct_test, preds_test, accuracy_test = GNN_trainer.evaluate(X, y_target, idx_test)
            results += [[epoch, loss, train_loss, accuracy_train, val_loss, accuracy_val, test_loss, accuracy_test]]

            if accuracy_val >= max_val_acc:
                # Update maximum validation accuracy
                max_val_acc = accuracy_val
                # store the state of the model when validation accuracy is maximum 
                state = dict([
                    ('model', copy.deepcopy(GNN_trainer.model.state_dict())), 
                    ('optim', copy.deepcopy(GNN_trainer.optimizer.state_dict()))])
            
            # Update stop_early to true if the Early stopping criterion is verified
            stop_early = Early_stopping_criterion.should_stop(epoch, val_loss, accuracy_val)

            epoch += 1
        return accuracy_val
    
    # Redefine evolutionary grid search fitness
    gs.fitness_individual = lambda x : fitness_GNN_individual(x, gs.grid_keys)
    # Search for the best configuration
    gs.algo()

    # Get the best individual
    unstring_list = lambda x: ast.literal_eval(x[4:]) # a convenient method to handle lists in grid search results 
    best_individual = [unstring_list(e) if (type(e) == str and e.startswith('LIST')) else e for e in gs.best_individual]

    print(f'{opt["dataset"]} {opt["model"]} {opt["split_name"]}: {best_individual} {gs.max_fitness}')
    # Store the best individual
    best_individuals.append(best_individual)

print(f'RESULTS: {opt["dataset"]} {opt["model"]} {split_suffix_}: {best_individuals}')




