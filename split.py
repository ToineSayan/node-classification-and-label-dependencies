import torch
import numpy as np
from sklearn.model_selection import train_test_split
from math import floor, ceil
import random
import load_datasets



# def train_val_test_split(
#     y_target,
#     splitting_method = 'fixed',
#     size_per_class = 20,
#     set_sizes = [140, 500, 1000],
#     dataset = '',
#     split_name = 'fixed'
#     ):
#     if splitting_method == 'random':
#         return split_random(y_target, set_sizes)
#     elif splitting_method == 'fixed':
#         return load_datasets.load_fixed_index(dataset.lower(), prefix = split_name)
#     elif splitting_method == 'random-stratified':
#         return split_stratified(y_target, set_sizes)
#     elif splitting_method == 'random-size-fixed':
#         return split_random_size_fixed(y_target, size_per_class, set_sizes)
#     elif splitting_method == 'pre-computed':
#         return load_pre_computed(dataset.lower(), split_name)

# # temporary
def train_val_test_split(
    splitting_method = 'pre-computed',
    dataset = '',
    split_name = 'planetoid'
    ):
    return load_pre_computed(dataset.lower(), split_name)

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
    
# print(load_pre_computed('pubmed-orig', 'split_6_3'))



# Returns 3 sets (train, val, test)
# set_sizes is a list of int of length 3 and defines [len(train set), len(val set), len(test set)]
# Each set is a random set of nodes and are disjoint
def split_random(y_target, set_sizes):
    num_of_nodes = y_target.shape[0]
    select = torch.randperm(num_of_nodes)
    print(select)
    idx_train, _ = torch.sort(select[0:set_sizes[0]])
    idx_val, _ = torch.sort(select[set_sizes[0]: set_sizes[0] + set_sizes[1]])
    idx_test, _ = torch.sort(select[set_sizes[0] + set_sizes[1]: set_sizes[0] + set_sizes[1] + set_sizes[2]])
    return(idx_train, idx_val, idx_test)

# Returns 3 sets (train, val, test)
# Set_sizes is a list of int of length 3 and defines [_, len(val set), len(test set)] (first value is ignored)
# The length of the train set is size_per_class*number_of_classes
# Validation and test sets are random sets of nodes and are disjoint
def split_random_size_fixed(y_target, size_per_class, set_sizes):
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
        for i in range(len(select)):
            idx_train[incr + i] = idx[select][i]
        incr += size_per_class
    idx_train, _ = torch.sort(idx_train, 0)

    # Second step: calculate idx_val and idx_test by selecting random indices in the list of indices 
    # that are not in idx_train (the remaining indices)
    select = torch.empty((num_of_nodes-len(idx_train)), dtype=torch.long)
    incr = 0
    idx_list_shuffled = torch.randperm(num_of_nodes)
    for i in idx_list_shuffled:
        if not i in idx_train:
            select[incr] = i
            incr += 1
    idx_val, _ = torch.sort(select[0:set_sizes[1]])
    idx_test, _ = torch.sort(select[set_sizes[1]: set_sizes[1] + set_sizes[2]])
    # print(y_target)
    # print(idx_train, idx_val, idx_test)
    return(idx_train, idx_val, idx_test)


def split_stratified(y_target, set_sizes):
    num_of_nodes = y_target.shape[0]
    classes = torch.unique(y_target) # Get the label values
     
    class_indices = [] # will be used to store the list of node indices per class
    class_indices_number = [] # will be used to store the number of nodes per class

    # Step 1: get the node indices and the number of nodes per class
    for c in classes:
        # Get all the indices of nodes with label c
        idx = (y_target == int(c)).nonzero(as_tuple=True)[0]
        class_indices.append(idx)
        class_indices_number.append(idx.shape[0])

    # # Step 2: calculate the number of nodes of each class to add in each set (training, validation, test)
    # sizes_float = []
    # class_repartition = []
    # sfs = []
    # # after the calculation of the number of node of each class
    # # we'll update the value of class_indices_number_ to remember the 
    # # remaining number of nodes available per class once the previous sets
    # # have been completed
    # class_indices_number_ = class_indices_number 
    # print(class_indices_number)
    # for i in range(len(set_sizes)):
    #     for j in range(len(class_indices_number)):
    #         n = class_indices_number[j]
    #         if n == 0:
    #             print("ERROR: empty class, cannot split")
    #             return None

    #         size_float = (set_sizes[i]*n)/num_of_nodes
    #         sfs.append(size_float) 
    #         print("aaaa")
    #         print(size_float)
    #         print(sfs)
    #         print(class_indices_number_)
    #         if size_float < 1:
    #             # 1st case: we need to have at least one node per class per set
    #             sizes_float.append({1}) 
    #         elif ceil(size_float) < class_indices_number_[j]:
    #             # 2nd case: the number of node per class per set is the floor or the
    #             # ceil value, but only if the ceil value is inferior to the number
    #             # of remaining nodes of this class
    #             sizes_float.append({floor(size_float), ceil(size_float)})
    #         elif floor(size_float) < class_indices_number_[j]:
    #             # 2nd case (bis): the number of node per class per set is the floor value, 
    #             # but only if the floor value is inferior to the number
    #             # of remaining nodes of this class
    #             sizes_float.append({floor(size_float)})
    #         else:
    #             print("ERROR: not enough nodes remaining (this ERROR will often raise if a class has less than 3 member nodes)")
    #             return None
    #     # print(class_indices_number_)
    #     # print(sizes_float)
    #     # print(sfs)
    #     sfs = []
    #     max_number = sum([max(i) for i in sizes_float])
    #     sizes_int = [min(i) for i in sizes_float]
    #     min_number = sum(sizes_int)
    #     adjustment = set_sizes[i] - min_number
    #     # print(f'adjustment: {adjustment}')
    #     # print(f'max number: {max_number}')
    #     if (max_number < set_sizes[i]) or (adjustment < 0):
    #         print("ERROR: adjustment impossible, cannot split")
    #         return None
    #     tmp = [i for i in range(len(sizes_float)) if len(sizes_float[i])>1]
    #     adjust_indices = random.sample(tmp, adjustment)
    #     for i in adjust_indices:
    #         sizes_int[i] = max(sizes_float[i])
    #     class_repartition.append(sizes_int)
    #     class_indices_number_ = [class_indices_number_[i] - sizes_int[i] for i in range(len(sizes_int))]
    #     sizes_float, sizes_int = [], []
    # # print(class_repartition)

    class_repartition = calculate_split_sizes_per_class(y_target, set_sizes)

    # Step 3: select indices for each set
    limits = [[0 for i in range(len(classes))]]
    for i in range(len(class_repartition)):
        tmp1 = class_repartition[i]
        tmp2 = limits[-1]
        limits.append([a+b for a,b in zip(tmp1, tmp2)])
    # Note: len(limits) = len(set_sizes) + 1
    tmp_indices = [torch.randperm(i) for i in class_indices_number] # len(tmp_indices) = len(classes)
    idx_sets = [torch.empty(set_sizes[i], dtype=torch.long) for i in range(len(set_sizes))]
    for i in range(len(set_sizes)):
        tmp = [tmp_indices[j][limits[i][j]:limits[i+1][j]] for j in range(len(classes))]
        list_indices = [class_indices[j][tmp[j]] for j in range(len(classes))]
        # print(list_indices)
        start = 0
        for j in range(len(classes)):
            for k in range(len(list_indices[j])):
                idx_sets[i][start + k] = list_indices[j][k]
            start = start + len(list_indices[j])
    
    # Step 4: sort indices
    idx_train, _ = torch.sort(idx_sets[0])
    idx_val, _ = torch.sort(idx_sets[1])
    idx_test, _ = torch.sort(idx_sets[2])
    return(idx_train, idx_val, idx_test)







def calculate_class_repartition(y_target, idx_sets):
    num_of_nodes = y_target.shape[0]
    classes = torch.unique(y_target) # Get the label values

    repartition = []
    for idx in idx_sets:
        r = []
        tmp = y_target[idx]
        for c in classes:
            # Get all the indices of nodes with label c
            tmp_idx = (tmp == int(c)).nonzero(as_tuple=True)[0]
            r.append(len(tmp_idx))
        repartition.append(r)
    return(classes, repartition)



def calculate_split_sizes_per_class(y_target, idx_sets):
    num_of_nodes = y_target.shape[0]
    classes = torch.unique(y_target) # Get the label values

    class_indices = [] # will be used to store the list of node indices per class
    class_indices_number = [] # will be used to store the number of nodes per class
    

    # Step 1: get the node indices and the number of nodes per class
    for c in classes:
        # Get all the indices of nodes with label c
        idx = (y_target == int(c)).nonzero(as_tuple=True)[0]
        class_indices.append(idx)
        class_indices_number.append(idx.shape[0])

    print("Proportions :")
    print([round(100*a/num_of_nodes)/100 for a in class_indices_number])

    sets_min = []
    deltas = []
    delta_vals = []
    for len_idx in idx_sets:
        tmp = [max(floor(len_idx*i/num_of_nodes),1) for i in class_indices_number]
        sets_min.append(tmp)
        tmp_delta = [len_idx*i/num_of_nodes - floor(len_idx*i/num_of_nodes) > 0 for i in class_indices_number]
        deltas.append(tmp_delta)
    delta_vals = [idx_sets[i] - sum(sets_min[i]) for i in range(len(idx_sets))] 
    class_reservoir = [class_indices_number[j] - sum([sets_min[i][j] for i in range(len(sets_min))]) for j in range(len(class_indices_number))]
    
    sets = sets_min
    complete = False
    while not complete:
        ind = [j for j in range(len(delta_vals)) if not delta_vals[j] == 0]
        if ind == []:
            complete = True
        else:
            random.shuffle(ind)
            i_selected = ind[0] # index of the a set to be completed
            try:
                tmp1 = [j for j in range(len(deltas[i_selected])) if deltas[i_selected][j]] # all the classes that miss an element
                tmp2 = [j for j in range(len(class_reservoir)) if class_reservoir[j] == 0] # available classes
                possible_classes = [a for a in tmp1 if a not in tmp2]
                class_selected = random.sample(possible_classes, 1)[0]
            except:
                tmp2 = [j for j in range(len(class_reservoir)) if class_reservoir[j] == 0] # available classes
                possible_classes = [a for a in range(len(class_indices_number)) if a not in tmp2]
                class_selected = random.sample(possible_classes, 1)[0]
            deltas[i_selected][class_selected] = False
            delta_vals[i_selected] = delta_vals[i_selected] - 1
            class_reservoir[class_selected] = class_reservoir[class_selected] - 1
            sets[i_selected][class_selected] = sets[i_selected][class_selected] + 1
    
    print("Sets proportions:")
    for i in sets:
        print([round(100*a/sum(i))/100 for a in i])
    return sets
        
            
    
    
    
    # print(sets_min)
    # print(deltas)
    # print(delta_vals)
    # print(class_reservoir)
    # # print(possible_indices)
    # print(sets)

# y = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# random.shuffle(y)
# y = torch.LongTensor(y)
# calculate_split_sizes_per_class(y, [10, 7, 3])

# print("bla")

# calculate_split_sizes_per_class(torch.randint(0,7,(1000,)), [300, 200, 500])





            

                

        



# split_random(torch.randint(0,7,(100,)), [10, 20, 50])
# split_random_size_fixed(torch.randint(0,7,(100,)), 3, [10, 20, 5])


# y = torch.randint(0,7,(100000,))
# idx_train_, idx_val_, idx_test_ = split_stratified(y, [20000,20000,10000])
# print('check')
# print(calculate_class_repartition(y, [idx_train_, idx_val_, idx_test_]))
# print(idx_train_, idx_val_, idx_test_)

# def get_train_idx(y_target, size_per_class):
#     # Calculate a train set with 20 nodes of each class
#     classes = torch.unique(y_target) # Get the label values
#     idx_tensor = torch.empty((size_per_class*len(classes)), dtype=torch.long)
#     incr = 0
#     for c in classes:
#         # Get all the indices of nodes with label c
#         idx = (y_target == int(c)).nonzero(as_tuple=True)[0]
#         # Select a random subset of 20 indices of idx
#         select = torch.randperm(len(idx))[0:size_per_class]
#         # Fill idx_train with the set of index extracted from idx
#         for i in range(len(select)):
#             idx[select][i]
#             idx_tensor[incr + i] = idx[select][i]
#         incr += size_per_class
#     idx_tensor, _ = torch.sort(idx_tensor, 0)
#     return idx_tensor

# def split_random_fixed(y_target, sizes_per_class):
#     # Calculate a train set with 20 nodes of each class
#     classes = torch.unique(y_target) # Get the label values
#     total = sum(sizes_per_class)
#     print(total)

#     train_tensors = set()
#     val_tensors = set()
#     test_tensors = set()
#     for c in classes:
#         # Get all the indices of nodes with label c
#         idx = (y_target == int(c)).nonzero(as_tuple=True)[0]
#         # Select a random subset of 20 indices of idx
#         print(len(idx))
#         select = torch.randperm(len(idx))[0:total]
#         # Fill idx_train with the set of index extracted from idx
#         (train_idx, val_idx, test_idx) = torch.split(select, sizes_per_class)
#         train_tensors.add(idx[train_idx])
#         val_tensors.add(idx[val_idx])
#         test_tensors.add(idx[test_idx])
#         print(idx[val_idx].shape)
    
#     idx_train, _  = torch.sort(torch.cat(tuple(train_tensors), 0))
#     idx_val, _  = torch.sort(torch.cat(tuple(val_tensors), 0))
#     idx_test, _  =  torch.sort(torch.cat(tuple(test_tensors), 0))
#     return(idx_train, idx_val, idx_test)



        





# def train_val_test_split(
#         idx_list, 
#         y_target = None, 
#         size_per_class = 20, 
#         train_size = 200, 
#         test_size = 1000, 
#         val_size = 500, 
#         method = 'random-fixed', 
#         train_set = None
#     ):
#     if method == 'random-stratify':
#         idx_train = idx_stratified_split(idx_list, y_target)
#     elif method == 'predefinite':
#         idx_train = train_set
#     elif method == 'random-fixed':
#         idx_train = get_train_idx(y_target, size_per_class)
#     elif method == 'random':
#         select = torch.randperm(len(idx_list))
#         idx_train = idx_list[select[0:train_size]]
#     tmp = torch.empty((len(idx_list)-len(idx_train)), dtype=torch.long)
#     incr = 0
#     for i in idx_list:
#         if not i in idx_train:
#             tmp[incr] = i
#             incr += 1
#     select = torch.randperm(len(tmp))
#     idx_test = tmp[select[0:test_size]]
#     idx_val = tmp[select[test_size:test_size + val_size]]
#     return(idx_train, idx_val, idx_test)

# def idx_stratified_split(idx_list, y_target, train_size):
#     targets = y_target.numpy()
#     indices, _ = train_test_split(
#         np.arange(len(targets)),
#         train_size = train_size,
#         stratify= targets
#     )
#     tmp = torch.LongTensor(indices)
#     idx_train, _ = torch.sort(torch.LongTensor(idx_list[tmp]), 0)
#     return idx_train


# def load_one_fixed(f_):
#     f = open(f_, 'r')
#     l = []
#     for line in f:
#         idx = int(line.strip())
#         l.append(idx)
#     idx_tensor = torch.LongTensor(l)
#     return idx_tensor

# def load_fixed_wikivitals():
#     idx_train = load_one_fixed('./datasets/wikivitals/sets/train_stratified.csv')
#     idx_val = load_one_fixed('./datasets/wikivitals/sets/val_stratified.csv')
#     idx_test = load_one_fixed('./datasets/wikivitals/sets/test_stratified.csv')
#     return(idx_train, idx_val, idx_test)



# def load_sets(dataset_str, set_type='fixed'):
#     SETS_DIR = './datasets/' + dataset_str + '/sets/'
#     idx_train = load_one_fixed(SETS_DIR + 'train_' + set_type + '.csv')
#     idx_val = load_one_fixed(SETS_DIR + 'val_' + set_type + '.csv')
#     idx_test = load_one_fixed(SETS_DIR + 'test_' + set_type + '.csv')
#     return(idx_train, idx_val, idx_test)

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

def split_k_folds(dataset, k):
    # outfile = open('./datasets/cora-orig/cora_splits.txt', 'w', encoding='utf8')
    outfile = open(f'./datasets/{dataset.lower()}/{dataset.lower()}_test_splits_sparse_strat.txt', 'w', encoding='utf8')
    X, y_target, _ = load_datasets.load_dataset(dataset)
    y_array = y_target.numpy()
    stkf = StratifiedKFold(k, shuffle=True, random_state = 42)
    cnt = 0
    for train, test in stkf.split(np.zeros(X.shape[0]), y_array):
        # test_str = '[' + ', '.join([str(j) for j in test]) + ']'
        test_l = sorted([int(j) for j in test])
        for i in range(1):
            train_i, val_i = train_test_split(train, test_size = 0.1, stratify = y_array[train], random_state = 42)
            _, small_train_i = train_test_split(train_i, test_size = 0.05, stratify = y_array[train_i], random_state = 42)
            _, sparse_strat_train_i = train_test_split(train_i, test_size = 640, stratify = y_array[train_i], random_state = 42)
            
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
                # print(train_20)
                # print(idx_20)
            indices_20 = np.sort(indices_20)
            print(len(indices_20))
            print(indices_20)
            # train_i_str = '[' + ', '.join([str(j) for j in train_i]) + ']'
            # val_i_str = '[' + ', '.join([str(j) for j in val_i]) + ']'
            train_i_l = sorted([int(j) for j in train_i])
            val_i_l = sorted([int(j) for j in val_i])
            small_train_i_l = sorted([int(j) for j in small_train_i])
            sparse_strat_train_i_l = sorted([int(j) for j in sparse_strat_train_i])

            if i == 0:
                # l = f'split_{cnt}_s\t{train_i_l}\t{val_i_l}\t{test_l}\n'
                # l += f'split_{cnt}_r\t{small_train_i_l}\t{val_i_l}\t{test_l}\n'
                # l += f'split_{cnt}_20\t{list(indices_20)}\t{val_i_l}\t{test_l}\n'
                l = f'split_{cnt}_spstrat\t{sparse_strat_train_i_l}\t{val_i_l}\t{test_l}\n'
            # else:
            #     l = f'split_{cnt}_{i-1}\t{train_i_l}\t{val_i_l}\t{test_l}\n'
            outfile.write(l)
        cnt += 1
    outfile.close()

if __name__ == "__main__":
    # split_k_folds('CORA-ORIG', 10)
    # split_k_folds('CITESEER-ORIG', 10)
    # split_k_folds('PUBMED-ORIG', 10)
    # split_k_folds('WIKIVITALS', 10)
    # split_k_folds('CITESEER-ORIG-LCC', 10)
    split_k_folds('WIKIVITALS_NEW', 10)


#    print(load_pre_computed('WIKIVITALS_NEW', 'split_0_r'))