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


# Method to transform data for Cora, Citeseer, Pubmed in a .npz file
def transform_original_data_to_npz(dataset_str):
    '''Transform original data for Cora, Citeseer, and Pubmed in a npz file
    
    The file is created in directory: './datasets/' + dataset_str + '-orig/'

    Parameters
    ----------
    dataset_str: str
        Name of the dataset (may be 'cora', 'citeseer', or 'pubmed')

    Returns
    -------
    None
    '''
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


    np.savez_compressed('./datasets/' + dataset_str + '-orig/' + dataset_str + '-orig.npz' , **d_npz)
    return None




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
    '''Load a dataset stored in a .npz file

    Parameters
    ----------
    dataset_str: str
        Name of the dataset (may be 'cora-orig', 'citeseer-orig', 'pubmed-orig', or 'wikivitals_new')
    
    attr: list of str
        List of attributes to concatenate. 'attr' by default if only one type of attributes.
    
    labels: str
        Name of the labels to use if multiple labellizations available

    Returns
    -------
    X: Tensor
        Node features. Shape is number of nodes * number of features
    
    y_target: LongTensor
        Target values (labels). Shape is number of nodes

    adj_tensor: Sparse Float Tensor
        Raw adjacency matrix. Shape is number of nodes * number of nodes
    '''
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



# def load_fixed_index(dataset_str, prefix = 'fixed'):
#     file = './datasets/' + dataset_str + '/' + dataset_str + '.npz'
#     dataset = np.load(file, allow_pickle=True)
#     # load train, val, test idx
#     train_idx = torch.LongTensor(dataset[prefix + '_train_set'])
#     val_idx = torch.LongTensor(dataset[prefix + '_val_set'])
#     test_idx = torch.LongTensor(dataset[prefix + '_test_set'])
#     return train_idx, val_idx, test_idx



# -------------------------------
# Adjacency matrix transformation
# -------------------------------

def fill_diag(sparseTensor, action):
    '''Fill the diagonal of a tensor with 1s or 0s

    Parameters
    ----------
    sparseTensor: Sparse Float Tensor
        The tensor whose diagonal has to be filled
    
    action: str
        'remove_self_links' to fill the diagonal with 0s.
        'add_self_links' to fill the diagonal with ones

    Returns
    -------
    sparseTensor: Sparse Float Tensor
        Modified tensor
    '''
    if not action == None:
        indices = sparseTensor._indices()
        idx = indices[0] != indices[1] # Calculate a mask to keep all indices except the ones on the diagonal
        indices = torch.vstack((indices[0][idx], indices[1][idx]))
        values = sparseTensor._values()[idx]
        tmp = torch.sparse.FloatTensor(indices, values, sparseTensor.shape)

        if action == 'remove_self_links':
            return tmp
        if action == 'add_self_links':
            In = torch.eye(sparseTensor.shape[0]).to_sparse()
            return In + tmp
    else:
        return sparseTensor

        

def eliminate_zeros(x):
    '''Remove zero in the list of values
    '''
    mask = x._values().nonzero()
    nv = x._values().index_select(0, mask.view(-1))
    ni = x._indices().index_select(1, mask.view(-1))
    return torch.sparse.FloatTensor(ni, nv, x.shape)

# Assumption: all values in adj_tsr ar 0 (no edge) or 1 (edge)
def to_symmetric_with_ones(adj_tsr, action_on_self_links = None):
    '''Transform a square matrix into a symmetric one (with self-links added or removed)

    Parameters
    ----------
    adj_tsr: Sparse Float Tensor
        The matrix to transform

    action_on_self_links: str
        'remove_self_links' to fill the diagonal with 0s.
        'add_self_links' to fill the diagonal with ones
    
    Returns
    -------
    adj_sym: Sparse Float Tensor
        Symmetric matrix
    '''
    adj_sym = adj_tsr + adj_tsr.t()
    adj_sym = fill_diag(adj_sym, action_on_self_links)
    adj_sym = adj_sym.coalesce() # to remove duplicate edges
    indices = adj_sym._indices()
    values = torch.ones(adj_sym._values().shape[0])
    adj_sym = torch.sparse.FloatTensor(indices, values, adj_tsr.shape)
    adj_sym = eliminate_zeros(adj_sym)
    adj_sym = adj_sym.coalesce() 
    
    return adj_sym


def calculate_degrees(adj, power = 1):
    """Calculation the degree matrix (at a given power)
    """
    degrees = torch.sparse.sum(adj, dim=1)
    if not power == 1:
        degrees = torch.pow(degrees, power)
    i = (degrees._indices()[0]).tolist()
    indices = torch.LongTensor([i, i])
    data = degrees._values()
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.sparse.FloatTensor(indices, data, adj.shape)


def transform_adjacency(adj, normalization_trick = "None", to_symmetric = False, action_on_self_links = None):
    """Transform the adjacency matrix

    If normalization_trick is set to any value in ["normalize_D-1", "normalize_D-0.5"] 
    has priority over the other parameters
    Else this method transforms (by default) the adjacency matrix into a symmetric adjacency 
    matrix with self links added
    """
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



