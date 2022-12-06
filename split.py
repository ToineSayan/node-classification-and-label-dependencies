import torch
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold
import random
import load_datasets



def sets_statistics(y_target, tuple_of_sets):
    print('Sets statistics:')
    num_of_nodes = y_target.shape[0]
    classes = torch.unique(y_target) # Get the label values
    for s in tuple_of_sets:
        set_size = len(s)
        y_reduced = y_target[s]
        store_stats = []
        for c in classes:
            # Get all the indices of nodes with label c
            idx = (y_reduced == int(c)).nonzero(as_tuple=True)[0]
            store_stats.append((int(c), len(idx)))
        print(f'Set size: {set_size}, Class repartition: {store_stats}')
    return None

            

def train_val_test_split(
    y_target = None,
    splitting_method = 'pre_computed',
    size_per_class = 20,
    set_sizes = [140, 500, 1000],
    dataset = 'CORA-ORIG',
    split_name = 'planetoid'
    ):
    if splitting_method == 'pre_computed':
        return load_pre_computed(dataset.lower(), split_name)
    elif splitting_method == 'random-with-stratified-train':
        return split_random_with_stratified_train(y_target, size_per_class, set_sizes)
    elif splitting_method == 'random-with-balanced-train':
        return split_random_with_balanced_train(y_target, size_per_class, set_sizes)
    elif splitting_method == 'random':
        return split_random(y_target, set_sizes)



# load pre-computed splits of a dataset
def load_pre_computed(dataset, split_name = 'split_0_s'):
    split_file = open(f'./datasets/{dataset}/{dataset}_splits.txt', 'r', encoding='utf8')
    for l in split_file:
        name, train, val, test = l.strip().split('\t')
        if name.strip() == split_name:
            break
    split_file.close()
    if not name.strip() == split_name:
        print(f'No pre-computed split whith name: {split_name}')
        return None
    idx_train = torch.LongTensor([int(i) for i in train.strip('[]').split(', ')])
    idx_val = torch.LongTensor([int(i) for i in val.strip('[]').split(', ')])
    idx_test = torch.LongTensor([int(i) for i in test.strip('[]').split(', ')])
    return(idx_train, idx_val, idx_test)
    



# Returns 3 disjoint sets of labelled nodes of sizes set_sizes (which must be a list of 3 integers)
# Each set is a random set of nodes and are disjoint
def split_random(y_target, set_sizes):
    num_of_nodes = y_target.shape[0]
    select = torch.randperm(num_of_nodes)
    print(select)
    idx_train, _ = torch.sort(select[0:set_sizes[0]])
    idx_val, _ = torch.sort(select[set_sizes[0]: set_sizes[0] + set_sizes[1]])
    idx_test, _ = torch.sort(select[set_sizes[0] + set_sizes[1]: set_sizes[0] + set_sizes[1] + set_sizes[2]])
    return(idx_train, idx_val, idx_test)


# Returns 3 disjoint sets of labelled nodes
# The first set is balanced and contains size_per_class nodes of each class in y_target
# The two other sets are a random selection of nodes among the remaing ones 
# with sizes set_sizes[-2:] (the last two values)
def split_random_with_balanced_train(y_target, size_per_class, set_sizes):
    num_of_nodes = y_target.shape[0]
    classes = torch.unique(y_target) # Get the label values
    
    # First step: calculate idx_train with size_per_class nodes of each class in it
    idx_train = torch.empty((size_per_class*len(classes)), dtype=torch.long)
    incr = 0
    for c in classes:
        # Get all the indices of nodes with label c
        idx = (y_target == int(c)).nonzero(as_tuple=True)[0]
        # Select a random subset of 20 indices of idx
        select = torch.randperm(len(idx))[0:size_per_class]
        idx_train[incr:incr+size_per_class] = idx[select]
        incr += size_per_class
    idx_train, _ = torch.sort(idx_train, 0)

    # Second step: calculate idx_val and idx_test by selecting random indices in the list of indices 
    # that are not in idx_train (the remaining indices)
    idx_remain = torch.LongTensor([i for i in range(num_of_nodes)])
    idx_remain = idx_remain[~idx_remain.unsqueeze(1).eq(idx_train).any(1)]
    select = torch.randperm(len(idx_remain))

    idx_val, _ = torch.sort(select[0:set_sizes[1]])
    idx_test, _ = torch.sort(select[set_sizes[1]: set_sizes[1] + set_sizes[2]])
    return(idx_train, idx_val, idx_test)


# Returns 3 disjoint sets of labelled nodes
# The first set is stratified.
# The two other sets are a random selection of nodes among the remaing ones 
# Set sizes are defined by set_sizes (a list of 3 integers)
def split_random_with_stratified_train(y_target, size_per_class, set_sizes):
    num_of_nodes = y_target.shape[0]
    classes = torch.unique(y_target) # Get the label values
    
    # First step: calculate idx_train with size_per_class nodes of each class in it

    idx_all = torch.LongTensor([i for i in range(num_of_nodes)])
    idx_train, _ = train_test_split(idx_all, train_size = set_sizes[0], stratify = y_target[idx_all])
    idx_train, _ = torch.sort(idx_train, 0)

    # Second step: calculate idx_val and idx_test by selecting random indices in the list of indices 
    # that are not in idx_train (the remaining indices)
    idx_remain = torch.LongTensor([i for i in range(num_of_nodes)])
    idx_remain = idx_remain[~idx_remain.unsqueeze(1).eq(idx_train).any(1)]
    select = torch.randperm(len(idx_remain))

    idx_val, _ = torch.sort(select[0:set_sizes[1]])
    idx_test, _ = torch.sort(select[set_sizes[1]: set_sizes[1] + set_sizes[2]])
    return(idx_train, idx_val, idx_test)




# Pre-computation of the splits
def split_k_folds(dataset, k):
    outfile = open(f'./datasets/{dataset.lower()}/{dataset.lower()}_splits.txt', 'w', encoding='utf8')
    X, y_target, _ = load_datasets.load_dataset(dataset)
    y_array = y_target.numpy()
    stkf = StratifiedKFold(k, shuffle=True, random_state = 42)
    cnt = 0
    for train, test in stkf.split(np.zeros(X.shape[0]), y_array):
        test_l = sorted([int(j) for j in test])
        for i in range(1):
            train_i, val_i = train_test_split(train, test_size = 0.1, stratify = y_array[train], random_state = 42)
            _, sparse_strat_train_i = train_test_split(train_i, test_size = 640, stratify = y_array[train_i], random_state = 42) # WARNING: 640 is for a specific dataset, CHANGE IT!
            
            unique_classes = np.unique(y_array[train_i])
            indices_20 = []
            print(unique_classes)
            for unique_class in unique_classes:
                tmp = y_array[train_i] == unique_class
                train_20 = tmp.nonzero()[0]
                # idx_20 = torch.randperm(len(list(train_20)))
                idx_20 = list(range(len(list(train_20))))
                random.shuffle(idx_20)
                idx_20 = idx_20[:20]
                indices_20 = indices_20 + list(train_i[train_20[idx_20]])
            indices_20 = np.sort(indices_20)
            print(len(indices_20))
            print(indices_20)
            train_i_l = sorted([int(j) for j in train_i])
            val_i_l = sorted([int(j) for j in val_i])
            small_train_i_l = sorted([int(j) for j in small_train_i])
            sparse_strat_train_i_l = sorted([int(j) for j in sparse_strat_train_i])

            if i == 0:
                l = f'split_{cnt}_s\t{train_i_l}\t{val_i_l}\t{test_l}\n'
                l += f'split_{cnt}_20\t{list(indices_20)}\t{val_i_l}\t{test_l}\n'
                l = f'split_{cnt}_spstrat\t{sparse_strat_train_i_l}\t{val_i_l}\t{test_l}\n'
            outfile.write(l)
        cnt += 1
    outfile.close()




if __name__ == "__main__":
    print("Dataset is Cora")
    X, y_target, adj = load_datasets.load_dataset('CORA-ORIG')

    print("Statistics for pre-computed split 0_s (dense and train set stratified)")
    train, val, test = train_val_test_split(
        splitting_method = 'pre-computed',
        split_name = 'split_0_s'
    )
    sets_statistics(y_target, (train, val, test))
    print('\n')

    print("Statistics for pre-computed split 0_20 (sparse with balanced train set)")
    train, val, test = train_val_test_split(
        splitting_method = 'pre-computed',
        split_name = 'split_0_20'
    )
    sets_statistics(y_target, (train, val, test))
    print('\n')

    print("Statistics for pre-computed split 0_spstrat (sparse with stratified train set)")
    train, val, test = train_val_test_split(
        splitting_method = 'pre-computed',
        split_name = 'split_0_spstrat'
    )
    sets_statistics(y_target, (train, val, test))
    print('\n')

    print("Statistics for pre-computed split planetoid")
    train, val, test = train_val_test_split(
        splitting_method = 'pre-computed',
        split_name = 'planetoid'
    )
    sets_statistics(y_target, (train, val, test))
    print('\n')

    print("Statistics for a randomly generated split with a balanced train set")
    train, val, test = train_val_test_split(
        y_target = y_target,
        splitting_method = 'random-with-balanced-train',
        size_per_class = 20,
        set_sizes = [None , 500, 1000]
    )
    sets_statistics(y_target, (train, val, test))
    print('\n')

    print("Statistics for a randomly generated split with a stratified train set")
    train, val, test = train_val_test_split(
        y_target = y_target,
        splitting_method = 'random-with-stratified-train',
        set_sizes = [140 , 500, 1000]
    )
    sets_statistics(y_target, (train, val, test))
    print('\n')