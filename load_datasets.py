
import torch
from torch.nn.functional import normalize
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np



def load_dataset(dataset, attr = ['attr'], labels= 'labels'):
    if dataset in ['CORA-ORIG', 'CITESEER-ORIG', 'PUBMED-ORIG']:
        X, y_target, adj = load_dataset_npz(dataset.lower(), attr, labels)     
    if dataset in ['WIKIVITALS_NEW']:
        X, y_target, adj = load_dataset_npz(dataset.lower(), attr = ['attr'], labels= 'labels_1')
    return(X, y_target, adj)  



def transform_original_data_to_npz(dataset_str):
    
    # Using Kipf method to get Cora, Citeseer, and Pubmed data (with the canonical split from Planetoid)
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels_ = load_data(dataset_str)

    d_npz = {}
    # Transform to npz
    print(f"check sizes {train_mask.shape} {val_mask.shape} {test_mask.shape}")
    # adj.tocsr()
    print(len(adj.data))
    d_npz['adj_data'] = adj.data
    d_npz['adj_indices'] = adj.indices 
    d_npz['adj_indptr'] = adj.indptr 
    d_npz['adj_shape'] = adj.shape
    

    print(type(features))
    features = features.tocsr()
    print(type(features))
    d_npz['attr_data'] = features.data
    d_npz['attr_indices'] = features.indices 
    d_npz['attr_indptr'] = features.indptr 
    d_npz['attr_shape'] = features.shape
    # print(f'Length of features: {len(features.data)}')

    y_train_coo = coo_matrix(y_train)
    y_val_coo = coo_matrix(y_val)
    y_test_coo = coo_matrix(y_test)

    labels = torch.zeros(adj.shape[0], dtype=int) 
    labels_ = coo_matrix(labels_)
    values = labels_.data
    indices = np.vstack((labels_.row, labels_.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = labels_.shape
    Y = torch.sparse.LongTensor(i, v, torch.Size(shape)).to_dense()
    labels = torch.argmax(Y, dim=1)
    print(type(labels_))
    print(labels_.shape)

    d_npz['labels'] = labels
        
    # Train, val, test sets
    d_npz['canonical_train_set'] = torch.arange(adj.shape[0])[train_mask]
    d_npz['canonical_val_set'] = torch.arange(adj.shape[0])[val_mask]
    d_npz['canonical_test_set'] = torch.arange(adj.shape[0])[test_mask]


    np.savez_compressed('./datasets/' + dataset_str + '-orig/' + dataset_str + '-orig3.npz' , **d_npz)







# Structure of .npz files:
# dataset = {
# 'adj_data' -> adjacency: list of ones (define the edges)
# 'adj_indices' -> adjacency: list of indices for the adjacency matrix
# 'adj_indptr' -> adjacency: list of index pointers for the adjacency matrix
# 'adj_shape' -> adjacency: shape of the adjacency matrix
#
# 'FEATURE_data' -> features: list of weights of the FEATUREs 
# 'FEATURE_indices' -> features: list of indices for the FEATURE matrix
# 'FEATURE_indptr' -> features: list of index pointers for the FEATURE matrix
# 'FEATURE_shape' -> features: shape of the FEATURE matrix
# Note: a list of feature has to be provided (default feature is 'attr' (generally if only one set of features))
# 
# 'LABELS' -> labels: LABELS of each node

# }
def load_dataset_npz(dataset_str, attr = ['attr'], labels= 'labels'):
    file = './datasets/' + dataset_str + '/' + dataset_str + '.npz'
    dataset = np.load(file, allow_pickle=True)
    # load adjacency matrix
    adj = csr_matrix((dataset['adj_data'], dataset['adj_indices'], dataset['adj_indptr']), dataset['adj_shape']).tocoo()
    indices = torch.LongTensor(np.vstack((adj.row, adj.col)))
    values = torch.FloatTensor(adj.data)
    adj_tensor = torch.sparse.FloatTensor(indices, values, adj.shape)
    adj_tensor = adj_tensor.coalesce()

    # load features 
    X = []
    for a in attr:
        attr_ = csr_matrix((dataset[a+'_data'], dataset[a+'_indices'], dataset[a+'_indptr']), dataset[a+'_shape']).tocoo()
        indices = torch.LongTensor(np.vstack((attr_.row, attr_.col)))
        values = torch.FloatTensor(attr_.data)
        x = torch.sparse.FloatTensor(indices, values, attr_.shape).to_dense()
        X.append(x)
    if len(X)==1:
        X = X[0]
    else:
        X = torch.cat(tuple(X), 1)

    # load labels
    unique_values, tmp = np.unique(dataset[labels], return_inverse=True)
    y_target = torch.LongTensor(tmp)

    print("----- DATA LOADED -----")
    print(f'#Edges: {len(adj_tensor.values())}')
    print(f'#Features: {X.shape[1]}')
    print(f'#Nodes: {X.shape[0]}')
    print(f'#Classes: {len(unique_values)}')

    return X, y_target, adj_tensor

def load_fixed_index(dataset_str, prefix = 'fixed'):
    file = './datasets/' + dataset_str + '/' + dataset_str + '.npz'
    dataset = np.load(file, allow_pickle=True)
    # load train, val, test idx
    train_idx = torch.LongTensor(dataset[prefix + '_train_set'])
    val_idx = torch.LongTensor(dataset[prefix + '_val_set'])
    test_idx = torch.LongTensor(dataset[prefix + '_test_set'])
    return train_idx, val_idx, test_idx



# -------------------------------
# Adjacency matrix transformation
# -------------------------------

# def fill_diag(sparseTensor, action):
#     if not action == None:
#         num_nodes = sparseTensor.shape[0]
#         In = torch.eye(num_nodes).to_sparse()
#         tmp = sparseTensor * In
#         tmp = sparseTensor - tmp # diagonal removed
#         if action == 'remove_self_links':
#             return tmp
#         if action == 'add_self_links':
#             return In + tmp
#     else:
#         return sparseTensor

def fill_diag(sparseTensor, action):
    if not action == None:
        indices = sparseTensor._indices()
        idx = indices[0] != indices[1] # Calculate a mask to keep all indices except the ones on the diagonal
        indices = torch.vstack((indices[0][idx], indices[1][idx]))
        values = sparseTensor._values()[idx]
        tmp = torch.sparse.FloatTensor(indices, values, sparseTensor.shape)

        # num_nodes = sparseTensor.shape[0]
        # In = torch.eye(num_nodes).to_sparse()
        # tmp = sparseTensor * In
        # tmp = sparseTensor - tmp # diagonal removed
        if action == 'remove_self_links':
            return tmp
        if action == 'add_self_links':
            In = torch.eye(sparseTensor.shape[0]).to_sparse()
            return In + tmp
    else:
        return sparseTensor
    # Remove self links (calculates A~)
        

def eliminate_zeros(x):
    mask = x._values().nonzero()
    nv = x._values().index_select(0, mask.view(-1))
    ni = x._indices().index_select(1, mask.view(-1))
    return torch.sparse.FloatTensor(ni, nv, x.shape)

# Assumption: all values in adj_tsr ar 0 (no edge) or 1 (edge)
def to_symmetric_with_ones(adj_tsr, action_on_self_links = None):
    adj_sym = adj_tsr + adj_tsr.t()
    adj_sym = fill_diag(adj_sym, action_on_self_links)
    adj_sym = adj_sym.coalesce() # to remove duplicate edges
    indices = adj_sym._indices()
    values = torch.ones(adj_sym._values().shape[0])
    adj_sym = torch.sparse.FloatTensor(indices, values, adj_tsr.shape)
    adj_sym = eliminate_zeros(adj_sym)
    adj_sym = adj_sym.coalesce() 
    print(f'#Undirected edges: {len(adj_sym.values())}')
    
    return adj_sym

# Calculation the degree matrix (at a given power)
def calculate_degrees(adj, power = 1):
    degrees = torch.sparse.sum(adj, dim=1)
    if not power == 1:
        degrees = torch.pow(degrees, power)
    i = (degrees._indices()[0]).tolist()
    indices = torch.LongTensor([i, i])
    data = degrees._values()
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sparse.FloatTensor(indices, data, adj.shape)

# Transform the adjacency matrix:
# If normalization_trick is set to any value in ["normalize_D-1", "normalize_D-0.5"] 
# has priority over the other parameters
# Else this method transforms (by default) the adjacency matrix into a symmetric adjacency 
# matrix with self links added
def transform_adjacency(adj, normalization_trick = "None", to_symmetric = False, action_on_self_links = None):
    if normalization_trick == "normalize_D-1":
        print('INFO: The normalization trick includes the symmetrization of the matrix and the \
            addition of self-links (and overwrites the corresponding parameters)')
        A_ = to_symmetric_with_ones(adj, 'add_self_links')
        D_ = calculate_degrees(A_, power=-1)
        return torch.sparse.mm(D_, A_)
    elif normalization_trick == "normalize_D-0.5":
        print('INFO: The normalization trick includes the symmetrization of the matrix and the \
            addition of self-links (and overwrites the corresponding parameters)')
        A_ = to_symmetric_with_ones(adj, 'add_self_links')
        D_ = calculate_degrees(A_, power=-0.5)
        tmp = torch.sparse.mm(A_, D_)
        return torch.sparse.mm(D_, tmp)
    elif normalization_trick == 'FAGCN_norm':
        A_ = to_symmetric_with_ones(adj, 'add_self_links')
        D_ = calculate_degrees(A_, power=-0.5)
        tmp = torch.sparse.mm(A_, D_)
        return torch.sparse.mm(D_, tmp)
    else:
        pass

    if to_symmetric:
        return to_symmetric_with_ones(adj, action_on_self_links)
    
    return adj





# -------------------------------
# Features pre-processing
# -------------------------------

# Binarization of features
def binarize_features(X, binarize = False):
    X_ = X
    if binarize:
        X_[X_ > 0] = 1
        print('Pre-processing: features binarized')
    return X_

# Normalization of features
def normalize_features(X, normalization = 'None'):
    if normalization == 'global-L1':
        print('Pre-processing: feature normalization using l1-norm.')
        return normalize(X, p=1, dim = 1)
    elif normalization == 'global-L2':
        print('Pre-processing: feature normalization using l2-norm.')
        return normalize(X)
    else:
        print('Pre-processing: features not normalized')
        return X

# chi-2 selection of features
def chi2_feature_selection(X, y_target, idx_train, chi2_selection = False, k_best_value = 100):
    if chi2_selection:
        reports.print_report_line('LOAD DATASET', 'Feature selection', {"method": "chi-squared", "type": "k-best", "number": k_best_value})
        # print(f"INFO: Extracting {k_best_value} best features by a chi-squared test")
        ch2 = SelectKBest(chi2, k=k_best_value)
        _ = ch2.fit_transform(X[idx_train], y_target[idx_train])
        X = ch2.transform(X)
        return torch.Tensor(X)
    else:
        return X

def get_largest_connected_component_(adj):
    data = adj.values().numpy()
    indices = adj.indices()
    rows, cols = indices[0].numpy(), indices[1].numpy()
    print(data)
    print(rows)
    adj_ = csr_matrix((data, (rows, cols)))

    # adj_ = adj.to_dense().numpy()
    adj_lcc, idx_lcc = get_largest_connected_component(adj_, return_labels=True)
    return torch.sparse.FloatTensor(adj_lcc), torch.LongTensor(idx_lcc)


def extract_llc(dataset_str, attr = ['attr'], labels= 'labels'):
    file = './datasets/' + dataset_str + '/' + dataset_str + '.npz'
    dataset = np.load(file, allow_pickle=True)
    collect = dict()
    # load adjacency matrix
    adj = csr_matrix((dataset['adj_data'], dataset['adj_indices'], dataset['adj_indptr']), dataset['adj_shape'])
    adj_lcc, idx_lcc = get_largest_connected_component(adj, return_labels=True)
    collect['adj_data'] = adj_lcc.data
    collect['adj_indices'] = adj_lcc.indices
    collect['adj_indptr'] = adj_lcc.indptr
    collect['adj_shape'] = np.array((len(idx_lcc), len(idx_lcc)))

    
    for a in attr:
        attr_ = csr_matrix((dataset[a+'_data'], dataset[a+'_indices'], dataset[a+'_indptr']), dataset[a+'_shape'])
        tmp_lcc = attr_[idx_lcc]
        # attr_data, attr_indices, attr_indptr = tmp.data, tmp.indices, tmp.indptr
        collect[a+'_data'] = tmp_lcc.data
        collect[a+'_indices'] = tmp_lcc.indices 
        collect[a+'_indptr'] = tmp_lcc.indptr
        collect[a+'_shape'] = np.array((len(idx_lcc),dataset[a+'_shape'][1]))
        print(len(tmp_lcc.data))
        print(collect[a+'_shape'])

    collect[labels] = dataset[labels][idx_lcc]

    print(f'number of edges: {len(adj_lcc.data)}')

    np.savez(dataset_str + '-lcc',**collect)
    return None



# -------------------------------
# Transform to npz
# -------------------------------

# Structure of .npz files:
# dataset = {
# 'adj_data' -> adjacency: list of ones (define the edges)
# 'adj_indices' -> adjacency: list of indices for the adjacency matrix
# 'adj_indptr' -> adjacency: list of index pointers for the adjacency matrix
# 'adj_shape' -> adjacency: shape of the adjacency matrix
#
# 'FEATURE_data' -> features: list of weights of the FEATUREs 
# 'FEATURE_indices' -> features: list of indices for the FEATURE matrix
# 'FEATURE_indptr' -> features: list of index pointers for the FEATURE matrix
# 'FEATURE_shape' -> features: shape of the FEATURE matrix
# Note: a list of feature has to be provided (default feature is 'attr' (generally if only one set of features))
# 
# 'LABELS' -> labels: LABELS of each node

# }

# parameter feature_int = True means the feature in the files are integers from 0 to features count - 1
def transform_to_npz(d_files, feature_int = True):
    
    print(f'Generation of the .npz file for dataset {d_files["name"]}')
    
    # 0 name of the npz
    npz_data = dict()

    # 1 get all node names and index them from 0 to num_nodes - 1
    # and get all label names and index them from 0 to num_labels - 1
    labels = [k for k in d_files.keys() if k.startswith('label')]
    attributes = [k for k in d_files.keys() if k.startswith('attr')]

    # 2 read labels file(s) (2 col : node name - label)
    # Create labels array
    vocab_nodes = dict()
    vocab_labels = dict()
    n_l = []
    node_idx, label_idx = 0, 0
    for label_key in labels:
        n_l_indices = []
        f = open(d_files[label_key], 'r')
        for line in f:
            node_name, label_name = line.strip().split('\t')
            n_l.append((node_name, label_name))
            n, l = vocab_nodes.get(node_name, -1), vocab_labels.get(label_name, -1)
            if n < 0:
                vocab_nodes[node_name] = node_idx
                node_idx += 1
            if l < 0:
                vocab_labels[label_name] = label_idx
                label_idx += 1
            n_l_indices.append((vocab_nodes[node_name], vocab_labels[label_name]))
        f.close()
        n_l_indices.sort()
        npz_data[label_key] = [l for _, l in n_l_indices]  
        print(len(n_l_indices))
        print(label_idx)

    print(f'{node_idx} nodes found')
    print(f'{label_idx} labels found')

    print(npz_data.keys())
       

    for attribute_key in attributes:
        # 3 read features file(s) (n cols...)
        # Create features index, indxptr, data
        f = open(d_files[attribute_key], 'r')
        set_of_features = set() 
        n_f_w_dict = dict() 
        vocab_features = dict() # dict features: id (id in range 0 to num. features - 1)
        for line in f:
            if line == '':
                break
            try:
                node_name, features_weights = line.strip().split('\t') 
                tmp = features_weights.strip().split()
            except:
                node_name = line.strip()
                features_weights = ''
                tmp = []
            n_f_w_dict[vocab_nodes[node_name]] = []
            for fw_string in tmp:
                feat, w = fw_string.strip().split(':')
                w = float(w)
                set_of_features.add(feat)
                n_f_w_dict[vocab_nodes[node_name]].append((feat,w))
        f.close()
        if not feature_int:
            features_count = len(set_of_features)
            list_of_features = list(set_of_features)
            # sort list of feature by alpha-numeric order 
            list_of_features.sort()
        else:
            features_count = max([int(s) for s in set_of_features]) + 1
            print(features_count)
            list_of_features = [str(i) for i in range(features_count)]
        for i in range(features_count):
            vocab_features[list_of_features[i]] = i

        
        attr_indptr = [0]
        attr_indices = []
        attr_data = []
        attr_shape = [node_idx, features_count]
        nodes_without_features = []
        for n in range(node_idx):
            f_w = n_f_w_dict.get(n, [])
            if f_w == []:
                nodes_without_features.append(n)
            attr_indptr.append(attr_indptr[-1] + len(f_w))
            for feat,w in f_w:
                attr_indices.append(vocab_features[feat])
                attr_data.append(w)
        npz_data[attribute_key + '_data'] =  attr_data
        npz_data[attribute_key + '_indices'] = attr_indices
        npz_data[attribute_key + '_indptr'] = attr_indptr
        npz_data[attribute_key + '_shape'] = attr_shape

        print(f'{features_count} features found')
        print('Checking some values:')
        print(f'Length of attr_data: {len(attr_data)}')
        print(f'Length of attr_indices: {len(attr_indices)}')
        print(f'Length of attr_indptr: {len(attr_indptr)} (= node number + 1 if OK)')
        print(f'attr_shape: {attr_shape} (= node number, num features if OK)')
        if not nodes_without_features == []:
            print(f'Number of nodes without features: {len(nodes_without_features)} ({100*len(nodes_without_features)/node_idx}%)')


    # 4 read net file (3 cols: start end weight)
    # Create features index, indxptr, data

    # 'adj_data' -> adjacency: list of ones (define the edges)
    # 'adj_indices' -> adjacency: list of indices for the adjacency matrix
    # 'adj_indptr' -> adjacency: list of index pointers for the adjacency matrix
    # 'adj_shape' -> adjacency: shape of the adjacency matrix

    f = open(d_files['net'], 'r')
    net_dict = dict()
    for i in range(node_idx):
        net_dict[i] = []
    for line in f:
        if not line.strip() == '':
            start, end, w = line.strip().split()
            u, v, w = vocab_nodes[start], vocab_nodes[end], float(w)
            net_dict[u].append((v, w))
    f.close()

    adj_indptr = [0]
    adj_indices = []
    adj_data = []
    adj_shape = [node_idx, node_idx]
    for i in range(node_idx):
        adj_indptr.append(adj_indptr[-1] + len(net_dict[i]))
        # Warning: 2 or more weights can be defined for the same edge (u,v)
        # Don't forget to coalesce after loading
        for v, w in net_dict[i]:
            adj_indices.append(v)
            adj_data.append(w)
    print(f'{len(adj_indices)} edges found')
    print('Checking some values:')
    print(f'Length of adj_data: {len(adj_data)}')
    print(f'Length of adj_indices: {len(adj_indices)}')
    print(f'Length of adj_indptr: {len(adj_indptr)} (= node number + 1 if OK)')
    print(f'adj_shape: {adj_shape} (= node number, node number if OK)')

    npz_data['adj_data'] =  adj_data
    npz_data['adj_indices'] = adj_indices
    npz_data['adj_indptr'] = adj_indptr
    npz_data['adj_shape'] = adj_shape
        

    np.savez(d_files['name'],**npz_data)

    return True

if __name__ == "__main__":
    print(load_fixed_index('cora-orig', 'canonical'))

