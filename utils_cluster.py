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