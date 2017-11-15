import numpy as np
from scipy.sparse import csr_matrix

def convert(data, size=68, mode = 'vec2mat'): #diag=0,
    '''
    Convert data from upper triangle vector to square matrix or vice versa
    depending on mode.

    INPUT : 

    data - vector or square matrix depending on mode
    size - preffered square matrix size (given by formula :
           (1+sqrt(1+8k)/2, where k = len(data), when data is vector)
    diag - how to fill diagonal for vec2mat mode
    mode - possible values 'vec2mat', 'mat2vec'

    OUTPUT : 

    square matrix or 1D vector 

    EXAMPLE :

    a = np.array([[1,2,3],[2,1,4],[3,4,1]])
    vec_a = convert(a, mode='mat2vec')
    print(vec_a)
    >>> array([2, 3, 4])

    convert(vec_a, size = 3, diag = 1, mode = vec2mat)
    >>> matrix([[1, 2, 3],
                [2, 1, 4],
                [3, 4, 1]], dtype=int64)

    '''

    if mode == 'mat2vec':
        
        mat = data.copy()
        rows, cols = np.triu_indices(data.shape[0],k = 0)
        vec = mat[rows,cols]
        
        return vec

    elif mode == 'vec2mat':
        
        vec = data.copy()        
        rows, cols = np.triu_indices(size,k = 0)
        mat = csr_matrix((vec, (rows, cols)), shape=(size, size)).todense()
        mat = mat + mat.T # symmetric matrix
        np.fill_diagonal(mat, np.diag(mat)/2)
        return mat