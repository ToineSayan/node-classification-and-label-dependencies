from scipy.sparse import csr_matrix, hstack, SparseEfficiencyWarning, diags
import numpy as np
import sklearn.preprocessing
import warnings 
warnings.simplefilter('ignore',SparseEfficiencyWarning)



def load_dataset(dataset, attr = ['attr'], labels= 'labels'):
    if dataset in ['CORA-ORIG', 'CITESEER-ORIG', 'PUBMED-ORIG']:
        X, y_target, adj = load_dataset_npz(dataset.lower(), attr, labels)     
    if dataset in ['WIKIVITALS_NEW']:
        X, y_target, adj = load_dataset_npz(dataset.lower(), attr = ['attr'], labels= 'labels_1')
    return(X, y_target, adj)  


# # Method to transform data for Cora, Citeseer, Pubmed in a .npz file
# def transform_original_data_to_npz(dataset_str):
#     '''Transform original data for Cora, Citeseer, and Pubmed in a npz file
    
#     The file is created in directory: './datasets/' + dataset_str + '-orig/'

#     Parameters
#     ----------
#     dataset_str: str
#         Name of the dataset (may be 'cora', 'citeseer', or 'pubmed')

#     Returns
#     -------
#     None
#     '''
#     # Using Kipf method to get Cora, Citeseer, and Pubmed data (with the canonical split from Planetoid)
#     adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels_ = load_data(dataset_str)

#     d_npz = {}
#     # Transform to npz
#     print(f"check sizes {train_mask.shape} {val_mask.shape} {test_mask.shape}")
#     # adj.tocsr()
#     print(len(adj.data))
#     d_npz['adj_data'] = adj.data
#     d_npz['adj_indices'] = adj.indices 
#     d_npz['adj_indptr'] = adj.indptr 
#     d_npz['adj_shape'] = adj.shape
    

#     print(type(features))
#     features = features.tocsr()
#     print(type(features))
#     d_npz['attr_data'] = features.data
#     d_npz['attr_indices'] = features.indices 
#     d_npz['attr_indptr'] = features.indptr 
#     d_npz['attr_shape'] = features.shape
#     # print(f'Length of features: {len(features.data)}')

#     y_train_coo = coo_matrix(y_train)
#     y_val_coo = coo_matrix(y_val)
#     y_test_coo = coo_matrix(y_test)

#     labels = torch.zeros(adj.shape[0], dtype=int) 
#     labels_ = coo_matrix(labels_)
#     values = labels_.data
#     indices = np.vstack((labels_.row, labels_.col))
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = labels_.shape
#     Y = torch.sparse.LongTensor(i, v, torch.Size(shape)).to_dense()
#     labels = torch.argmax(Y, dim=1)
#     print(type(labels_))
#     print(labels_.shape)

#     d_npz['labels'] = labels
        
#     # Train, val, test sets
#     d_npz['canonical_train_set'] = torch.arange(adj.shape[0])[train_mask]
#     d_npz['canonical_val_set'] = torch.arange(adj.shape[0])[val_mask]
#     d_npz['canonical_test_set'] = torch.arange(adj.shape[0])[test_mask]


#     np.savez_compressed('./datasets/' + dataset_str + '-orig/' + dataset_str + '-orig.npz' , **d_npz)
#     return None





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
    X: csr_matrix
        Node features. Shape is number of nodes * number of features
    
    y_target: numpy array
        Target values (labels). Shape is number of nodes

    adj: csr_matrix
        Raw adjacency matrix. Shape is number of nodes * number of nodes
    '''
    file = './datasets/' + dataset_str + '/' + dataset_str + '.npz'
    dataset = np.load(file, allow_pickle=True)
    # load adjacency matrix
    adj = csr_matrix((dataset['adj_data'], dataset['adj_indices'], dataset['adj_indptr']), dataset['adj_shape'])

    # load features 
    X = []
    for a in attr:
        attr_ = csr_matrix((dataset[a+'_data'], dataset[a+'_indices'], dataset[a+'_indptr']), dataset[a+'_shape'])
        X.append(attr_)
    if len(X)==1:
        X = X[0]
    else:
        X = hstack(X, format='csr')

    # load labels
    unique_values, y_target = np.unique(dataset[labels], return_inverse=True)

    print("----- DATA LOADED -----")
    print(f'#Edges: {len(adj.data)}')
    print(f'#Features: {X.shape[1]}')
    print(f'#Nodes: {X.shape[0]}')
    print(f'#Classes: {len(unique_values)}')

    return X, y_target, adj





# -------------------------------
# Adjacency matrix transformation
# -------------------------------

# Assumption: all values in adj_tsr ar 0 (no edge) or 1 (edge)
def to_symmetric_with_ones(adj, action_on_self_links = None):
    '''Transform a square matrix into a symmetric one (with self-links added or removed)

    Parameters
    ----------
    adj_tsr: csr_matrix
        The matrix to transform

    action_on_self_links: str
        'remove_self_links' to fill the diagonal with 0s.
        'add_self_links' to fill the diagonal with ones
    
    Returns
    -------
    adj_sym: csr_matrix
        Symmetric matrix
    '''
    # to symmetric
    rows, cols = adj.nonzero() 
    rows_sym = np.hstack([rows, cols])
    cols_sym = np.hstack([cols, rows])
    n = len(cols_sym)
    adj_sym = csr_matrix((np.ones(n), (rows_sym, cols_sym)), shape=adj.shape)
    adj_sym.sum_duplicates() # to remove duplicate couple of indices
    adj_sym[adj_sym > 0] = 1  # adjacency matrix has only 1s and 0s
 
    # add self links or remove them or do nothing
    if action_on_self_links == 'add_self_links':
        adj_sym.setdiag(1)
    elif action_on_self_links == 'remove_self_links':
        adj_sym.setdiag(0)
    else: 
        pass
    
    print('A_, the symmetric version of A, has been computed')
    return adj_sym



def calculate_degrees(adj, power = 1):
    """Calculation the degree matrix (at a given power)
    """
    tmp = np.squeeze(np.asarray(adj.sum(axis = 1)))
    degrees  = diags(tmp, format='csr')
    return degrees.power(power)





def transform_adjacency(adj, normalization_trick = "None", to_symmetric = False, action_on_self_links = None):
    """Transform the adjacency matrix

    If normalization_trick is set to any value in ["normalize_D-1", "normalize_D-0.5"] 
    has priority over the other parameters
    Else this method transforms (by default) the adjacency matrix into a symmetric adjacency 
    matrix with self links added
    """
    if normalization_trick == "normalize_D-1":
        # INFO: The normalization trick includes the symmetrization of the matrix and the
        # addition of self-links (and overwrites the corresponding parameters
        A_ = to_symmetric_with_ones(adj, 'add_self_links')
        D_ = calculate_degrees(A_, power=-1)
        print('A~ normalized: A~ = D^(-1)A_')
        return D_.dot(A_)
    elif normalization_trick == "normalize_D-0.5":
        # INFO: The normalization trick includes the symmetrization of the matrix and the
        # addition of self-links (and overwrites the corresponding parameters
        A_ = to_symmetric_with_ones(adj, 'add_self_links')
        D_ = calculate_degrees(A_, power=-0.5)
        print('A_ normalized: A~ = D^(-1/2)A_D^(-1/2)')
        return D_.dot(A_.dot(D_))
    else:
        
        pass

    if to_symmetric:
        return to_symmetric_with_ones(adj, action_on_self_links)
        print('A_ not normalized')
    
    print('A not normalized')    
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



def normalize_features(X, normalization = 'None'):
    if normalization == 'global-L1':
        print('Pre-processing: feature normalization using l1-norm.')
        return sklearn.preprocessing.normalize(X, norm='l1', axis = 1)
    elif normalization == 'global-L2':
        print('Pre-processing: feature normalization using l2-norm.')
        return sklearn.preprocessing.normalize(X, norm='l2', axis = 1)
    else:
        print('Pre-processing: features not normalized')
        return X




if __name__ == "__main__":
    from datetime import datetime

    # dataset = 'CORA-ORIG'
    dataset = 'WIKIVITALS_NEW'
    ts0  = datetime.timestamp(datetime.now())

    X, y_target, adj = load_dataset(dataset)
    X = binarize_features(X, True)
    X = normalize_features(X, 'global-L1')
    adj_norm = transform_adjacency(
        adj, 
        'normalize_D-0.5',
        to_symmetric = True, 
        action_on_self_links = 'add_self_links'
    )

    ts1  = datetime.timestamp(datetime.now())
    print(ts1-ts0)


