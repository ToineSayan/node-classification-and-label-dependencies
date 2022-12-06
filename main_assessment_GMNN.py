import torch
import json
import load_datasets
from split import train_val_test_split

from GMNN import GMNN

from scipy.stats import ttest_rel
import numpy as np
from math import floor, log10

import argparse
from statistics import mean, stdev

# A method to calculate a t-test of the null hypothesis between 2 distributions
# Alternative hypothesis to the null hypothesis is that the mean of the distribution
# underlying the first sample is less than the mean of the distribution underlying the second sample
def ttest(l1, l2):
    _, pval = ttest_rel(l1, l2, alternative='less')
    pval_ = round(pval, -int(floor(log10(abs(pval)))))
    if pval > 0.05:
        s = ''
    if pval <= 0.05:
        s = '*'
    if pval <= 0.01:
        s = '**'
    if pval <= 0.001:
        s = '***'
    return(f'{s} (p-value: {pval_})')


# ----- GET ARGS -----
# The only argument is the name of the configuration file to use where all
# the parameters of the training to be executed are defined.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CORA-ORIG')
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--split_suffix', type=str, default='s')
parser.add_argument('--strategy', type=str, default='identity') # 'identity', 'default_constant', 'best_config_p'
args = parser.parse_args()

dataset_ = vars(args)['dataset']
model_ = vars(args)['model']
split_suffix_ = vars(args)['split_suffix']
strategy_ = vars(args)['strategy']

# ----- CHECK IF CUDA AVAILABLE -----
torch.cuda.empty_cache()
cuda = torch.cuda.is_available()

# ----- LOAD CONFIGURATIONS -----
opt = json.load(open(f'./configs/assessment/assessment_GMNN_{model_}.json', 'r', encoding='utf8'))
opt['dataset'] = dataset_
opt['model'] = model_
opt['EM_loops'] = 1
print(opt)

# ----- LOAD THE BEST CONFIGURATIONS OF HYPERPARAMETERS CALCULATED
model_selection_results = json.load(open(f'./configs/grid_search/model_selection_results.json', 'r', encoding='utf8'))
tmp = model_selection_results[dataset_][model_]["split_suffix"][split_suffix_] 
tmp = [a + [dataset_, model_, split_suffix_] for a in tmp]
tmp2 = model_selection_results[dataset_][model_]["keys"] + ['dataset', 'model', 'split_suffix']
best_configurations = [dict(zip(tmp2, tmp[i])) for i in range(len(tmp))]

# ----- LOAD DATASET -----
# Loading and pre-processing of data
X, y_target, adj = load_datasets.load_dataset(opt['dataset'])
# Feature binarization (X unchanged if opt['binarize'] == False)
X = load_datasets.binarize_features(X, opt['binarize'])
# Feature normalization (X unchanged if opt['normalization'] == "None")
X = load_datasets.normalize_features(X, opt['normalization'])
# adjacency matrix normalization
adj_norm = load_datasets.transform_adjacency(
    adj, 
    opt['normalization_trick'],
    opt['to_symmetric'],
    opt['add_self_links']
    )

# Transform to tensors
tmp = X.tocoo()
indices = torch.LongTensor(np.vstack((tmp.row, tmp.col)))
values = torch.FloatTensor(tmp.data)
X = torch.sparse.FloatTensor(indices, values, tmp.shape).to_dense()

y_target = torch.LongTensor(y_target)

tmp = adj_norm.tocoo()
indices = torch.LongTensor(np.vstack((tmp.row, tmp.col)))
values = torch.FloatTensor(tmp.data)
adj_norm = torch.sparse.FloatTensor(indices, values, tmp.shape)

# One hot enconding of y_target 
y_target_bin = torch.nn.functional.one_hot(y_target)
num_classes = torch.unique(y_target).shape[0]
num_nodes, num_features = X.shape
opt['num_nodes'], opt['num_features'], opt['num_classes'] = num_nodes, num_features, num_classes




init_results = [] # will be used to store the results of the pre-training phase
final_results = [] # will be used to store the results of EM-loops phase
q_results, p_results = [],[] # will be used to store the results for GNNq and GNNp during EM-loops

print(opt)

for k in range(opt["k_num_splits"]): # for all pre-computed split...
    print(f'split {k}')
    opt['split_name'] = f'split_{k}_{split_suffix_}' # update split name (reconstruct it)
    for key, val in best_configurations[k].items(): opt[key] = val # use a pre-calculated best configuration for this split and this architecture

    # Get the list of indices for train, validation, and test sets
    idx_all = torch.LongTensor([i for i in range(opt['num_nodes'])])
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
        X = X.cuda()
        y_target = y_target.cuda()
        adj_norm = adj_norm.cuda()
        idx_all = idx_all.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()


    for i in range(opt['iterations']): # do R intializations of the model and evaluate it (R = opt['iterations'])
        opt['seed'] = i+1

        # Select a strategy to initialize and train GNNp (phi = theta (identity) if strategy is not specified)
        if strategy_ == 'default_constant': # use a default config for GNNp (same for all splits and datasets)
            opt_p = {
                "input_dropout": 0.5,
                "dropout": 0.5,
                "lr": 0.05,
                "hidden_dims": [16],
                "decay": 5e-4
                }
        elif strategy_ == 'identity': # use the same calculated for GNNq to initialize GNNp
            opt_p = None
        elif strategy_ == 'best_config_p': # use the best configuration of hyperparameters calculated for GNNp
            opt_p = dict()
            model_selection_results_GNNp = json.load(open(f'./configs/grid_search/model_selection_results_GNNp.json', 'r', encoding='utf8'))
            tmp = model_selection_results_GNNp[dataset_][model_]["split_suffix"][split_suffix_][k]
            tmp = tmp + [dataset_, model_, split_suffix_]
            tmp2 = model_selection_results[dataset_][model_]["keys"] + ['dataset', 'model', 'split_suffix']
            opt_p = dict(zip(tmp2, tmp))
        else:
            print(strategy_)
            print('Strategy not implemented')
            quit()



        GMNN_ = GMNN(opt, adj_norm, opt_p) # init GMNN
        pretrain_ = GMNN_.pretrain_q(X, y_target, idx_train, idx_val, idx_test) # pretrain GMNN
        init_results.append(pretrain_["test"])
        if opt['GMNN_architecture']: # GMNN is not activated for MLP architecture is the config. file
            final_ = GMNN_.do_EM_phases(X, y_target, idx_train, idx_val, idx_test, idx_all)
            print(f'test acc.: {100*pretrain_["test"]} --> {100*max([final_["test_p"], final_["test_q"]])}')
            final_results.append(max([final_["test_p"], final_["test_q"]]))
            q_results.append(final_["test_q"])
            p_results.append(final_["test_p"])
        else:
            print(f'test acc.: {100*pretrain_["test"]}')

if not len(init_results) <= 1:
    if not final_results == []:
        print(f'FINAL RESULTS: Means: {100*mean(init_results):.2f}% ({100*stdev(init_results):.2f}) --> {100*mean(final_results):.2f}% ({100*stdev(final_results):.2f}) (p: {100*mean(p_results):.2f}% ({100*stdev(p_results):.2f}), q: {100*mean(q_results):.2f}% ({100*stdev(q_results):.2f})')
        init_grouped = [mean(init_results[(i*opt['iterations']):((i+1)*opt['iterations'])]) for i in range(opt["k_num_splits"])]
        final_grouped = [mean(final_results[(i*opt['iterations']):((i+1)*opt['iterations'])]) for i in range(opt["k_num_splits"])]
        print(f'FINAL RESULTS: {zip(init_grouped, final_grouped)}')
        print(f'FINAL RESULTS: Delta: {100*mean(final_results) - 100*mean(init_results)} - significance: {ttest(init_grouped, final_grouped)})')
        print(f'FINAL RESULTS: exp3 (GMNN w. features) {opt["dataset"], opt["model"], opt["split_suffix"]}')
    else:
        print(f'FINAL RESULTS: Means: {100*mean(init_results):.2f}% ({100*stdev(init_results):.2f})')
        print(f'FINAL RESULTS: exp3 (MLP) {opt["dataset"], opt["model"], opt["split_suffix"]}')