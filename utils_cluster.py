import numpy as np
import copy

def genNormalizedX(*kinematics):
    '''
    The function aims to generate a numpy ndarray that contain kinematics information for clustering

    V_l: LV velocity. m*n nparray, m: number of trajectory, n: number of timestamp (length)
    V_f: FV velocity. m*n nparray, m: number of trajectory, n: number of timestamp (length)
    D: Distance between LV and FV. m*n nparray, m: number of trajectory, n: number of timestamp (length)
    '''
    # X = genX(V_l,V_f,D)
    # X_train = normalize(X)
    X, dimen = genX(kinematics)
    X_train = normalize(X,dimen)

    return X_train

# def genX(V_l,V_f,D):
#     '''
#     The function aims to generate a numpy ndarray that contains kinematics information

#     V_l: LV velocity. m*n nparray, m: number of trajectory, n: number of timestamp (length)
#     V_f: FV velocity. m*n nparray, m: number of trajectory, n: number of timestamp (length)
#     D: Distance between LV and FV. m*n nparray, m: number of trajectory, n: number of timestamp (length)
#     '''
#     X = np.zeros((V_l.shape[0],3*V_l.shape[1],2))
#     X[:,:V_l.shape[1],0], X[:,:V_l.shape[1],1] = copy.deepcopy(V_l), copy.deepcopy(V_l)
#     X[:,V_l.shape[1]:2*V_l.shape[1],0], X[:,V_l.shape[1]:2*V_l.shape[1],1] = copy.deepcopy(V_f), copy.deepcopy(V_f)
#     X[:,2*V_l.shape[1]:,0], X[:,2*V_l.shape[1]:,1] = copy.deepcopy(D), copy.deepcopy(D)
#     return X

def genX(kinematics):
    '''
    The function aims to generate a numpy ndarray that contains kinematics information

    example of kinematics:
    V_l: LV velocity. m*n nparray, m: number of trajectory, n: number of timestamp (length)
    V_f: FV velocity. m*n nparray, m: number of trajectory, n: number of timestamp (length)
    D: Distance between LV and FV. m*n nparray, m: number of trajectory, n: number of timestamp (length)
    '''
    dimen = len(kinematics) ## how many dimensions the kinematics are (the number of input kinematics)
    length = kinematics[0].shape[1] ## the length of the kinematics
    n = kinematics[0].shape[0] ## the number of the cases
    X = np.zeros((n,dimen*length,2))
    for i, kinematic in enumerate(kinematics):
        X[:,i*length:(i+1)*length,0], X[:,i*length:(i+1)*length,1] = copy.deepcopy(kinematic), copy.deepcopy(kinematic)
    return X, dimen

def normalize(X_train,dimen):
    '''
    The function aims to normalize the numpy nd array for clustering
    '''
    length = int(X_train.shape[1]/dimen)
    Mean, Std = [], []
    for i in range(dimen):
        mean = np.mean(X_train[:,i*length:(i+1)*length,0],axis=1).reshape(-1,1)
        std = np.std(X_train[:,i*length:(i+1)*length,0],axis=1).reshape(-1,1)
        std[std==0] = 1
        Mean.append(mean)
        Std.append(std)
    for i in range(dimen):
        X_train[:,i*length:(i+1)*length,1] = (X_train[:,i*length:(i+1)*length,0] - Mean[i])/Std[i]

    Mean, Std = [], []
    for i in range(dimen):
        mean = np.mean(X_train[:,i*length:(i+1)*length,1],axis=1).reshape(-1,1)
        std = np.std(X_train[:,i*length:(i+1)*length,1],axis=1).reshape(-1,1)
        std[std==0] = 1
        Mean.append(mean)
        Std.append(std)
    for i in range(dimen):
        X_train[:,i*length:(i+1)*length,1] = (X_train[:,i*length:(i+1)*length,1] - Mean[i])/Std[i]

    return X_train

# def normalize(X_train):
#     '''
#     The function aims to normalize the numpy nd array for clustering
#     '''
#     l = int(X_train.shape[1]/3)
#     mean1, mean2, mean3 = np.mean(X_train[:,:l,0],axis=1).reshape(-1,1), np.mean(X_train[:,l:l*2,0],axis=1).reshape(-1,1), np.mean(X_train[:,l*2:,0],axis=1).reshape(-1,1)
#     std1, std2, std3 = np.std(X_train[:,:l,0],axis=1).reshape(-1,1), np.std(X_train[:,l:l*2,0],axis=1).reshape(-1,1), np.std(X_train[:,l*2:,0],axis=1).reshape(-1,1)
#     std1[std1==0], std2[std2==0], std3[std3==0] = 1, 1, 1
#     X_train[:,:l,1] = (X_train[:,:l,0] - mean1)/std1
#     X_train[:,l:l*2,1] = (X_train[:,l:l*2,0] - mean2)/std2
#     X_train[:,l*2:,1] = (X_train[:,l*2:,0] - mean3)/std3
    
#     mean1, mean2, mean3 = np.mean(X_train[:,:l,1],axis=1).reshape(-1,1), np.mean(X_train[:,l:l*2,1],axis=1).reshape(-1,1), np.mean(X_train[:,l*2:,1],axis=1).reshape(-1,1)
#     std1, std2, std3 = np.std(X_train[:,:l,1],axis=1).reshape(-1,1), np.std(X_train[:,l:l*2,1],axis=1).reshape(-1,1), np.std(X_train[:,l*2:,1],axis=1).reshape(-1,1)
#     std1[std1==0], std2[std2==0], std3[std3==0] = 1, 1, 1   
#     X_train[:,:l,1] = (X_train[:,:l,1] - mean1)/std1
#     X_train[:,l:l*2,1] = (X_train[:,l:l*2,1] - mean2)/std2
#     X_train[:,l*2:,1] = (X_train[:,l*2:,1] - mean3)/std3
    
#     return X_train


# !pip install pycuda
# !pip install pyopencl
# !pip install GPUDTW

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, cdist
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

def hc_dm(X, alg="euclidean"):
    '''
    Calculate the distance matrix for H Clustering
    '''
    if alg == 'euclidean':
        dist_matrix = pdist(X)
    elif alg == 'fastdtw':
        from fastdtw import fastdtw
        dist_matrix = pdist(X, lambda u, v: fastdtw(u, v)[0])
    elif alg == 'dtw':
        from GPUDTW import cuda_dtw
        X = X.astype(np.float32)
        dist_matrix = cuda_dtw(X, X)
        dist_matrix = squareform(dist_matrix)
    else:
        raise ValueError("Invalid algorithm. Please choose 'euclidean' or 'fastdtw' or 'dtw'.")
    return dist_matrix

def hc_fcluster(dist_matrix, method='single', n_cls=16, figsize=(10, 8)):
    '''
    inputs: distance matrix
    output: cluster label
    '''
    ## form linkage
    Z = linkage(dist_matrix, method=method)
    ## assign labels
    y = fcluster(Z, t=n_cls, criterion='maxclust')
    ## plot dendrogram
    if figsize is not None:
        plt.figure(figsize=figsize)
        dendrogram(Z, truncate_mode='lastp', p=n_cls)
        plt.show()
    return y

def hc_secondary(dist_matrix,y,y_target,n_target,filter_threshold=None,method='single',figsize=(4,3.2)):
    '''
    dist_matrix: raw dm
    y: cluster labels
    y_target: the cluster label that we want to do secondary clustering
    n_target: number of clusters for secondary clustering
    '''
    ## extract distance matrix for
    dm_squareform = squareform(dist_matrix)
    dm_sec = dm_squareform[y==y_target][:,y==y_target]
    dm_sec = squareform(dm_sec)

    y_updated = np.copy(y)
    if figsize is None:
        y_sec = hc_fcluster(dm_sec, method=method, n_cls=n_target, figsize=None)
    else:
        y_sec = hc_fcluster(dm_sec, method=method, n_cls=n_target, figsize=figsize)
    if filter_threshold is not None:
        y_sec = filterOutlier(y_sec,threshold=filter_threshold)
    y_updated[y_updated==y_target] += y_sec
    return y_updated

def filterOutlier(y,threshold=2):
    '''
    threshold: number of minimum samples in a cluster
    '''
    y_updated = np.copy(y)
    labels, numbers = np.unique(y,return_counts=True)
    for i, label in enumerate(labels):
        if numbers[i]<threshold:
            y_updated[y==label] = 0
    return y_updated

def mergeOutlierToOther(dist_matrix,y,y_target):
    dm_sqr = squareform(dist_matrix)
    dm_sqr_target = dm_sqr[y==y_target]
    labels = np.unique(y)
    min_dist = np.zeros((dm_sqr_target.shape[0],labels.shape[0]))
    for i_label, label in enumerate(labels):
        min_dist[:,i_label] = np.min(dm_sqr_target[:,y==label],axis=1)
    y[y==y_target] = labels[labels!=y_target][np.argmin(min_dist[:,labels!=y_target],axis=1)]