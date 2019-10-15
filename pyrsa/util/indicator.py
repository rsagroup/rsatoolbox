import numpy as np 
'''
Collection of different kinds of indicator Matrices 
    identity: One column per unique element in vector  
    identity_pos: One column per unique non-zero element 
    allpairs:     All K*(K-1)/2 pairwise contrasts 
'''

def identity(c):
    """ Indicator matrix with one column per unique element in vector  
        Args:
            c (numpy.ndarray): n_row vector to code - discrete values (one dimensional)
        Returns:
            X (numpy.ndarray): n_row x n_values indicator matrix 
    """
    cc = np.unique(c)
    K = cc.size
    rows = np.size(c)
    
    X=np.zeros((rows,K))
    for i in range(K):
        X[c==cc[i],i]=1
    return X

def identity_pos(c):
    """ Indicator matrix with one column per unique positive element in vector  
        Args:
            c (numpy.ndarray): n_row vector to code - discrete values (one dimensional)
        Returns:
            X (numpy.ndarray): n_row x n_values indicator matrix 
    """
    cc = np.unique(c)
    K = cc.size
    rows = np.size(c)
    cc = cc[cc>0]
    K = cc.size
    X = np.zeros((rows,K))   
    for i in range(K):
        X[c==cc[i],i]=1
    return X 


def allpairs(c): 
    """ Indicator matrix with one row per unqiue pair 
        Args:
            c (numpy.ndarray): n_row vector to code - discrete values (one dimensional)
        Returns:
            X (numpy.ndarray): n_values*(n_values-1)/2 x n_row contrast matrix 
    """
    cc = np.unique(c)
    K = cc.size
    rows = np.size(c)
    X=np.zeros((int(K*(K-1)/2),rows))
    k=0
    # Now make a matrix with a pair of conditions per row 
    for i in range(K):
        for j in np.arange(i+1,K):
            X[k,c==cc[i]] = 1./sum(c==i)
            X[k,c==cc[j]] = -1./sum(c==j)
            k         = k+1
    return X

