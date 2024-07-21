import numpy as np
import copy
import random

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def LVActionPred(model,Input,mode,device):
    acc_min, acc_max, acc_interval = -9.5, 7, 0.25
    Acc_discrete = np.arange(acc_min, acc_max + acc_interval, acc_interval).round(2).tolist()
    
    input_tensor = torch.tensor(Input, dtype=torch.float32).to(device)
    p_a = model(input_tensor).to(device)
    p_a = torch.exp(p_a[:,-1,:])
    
    A_output = np.zeros(Input.shape[0])
    for i in range(A_output.shape[0]):
        p_a_i = p_a[i].tolist() # output of instance i
        if mode == "rs": ## Randomly sample (discrete action)
            a = random.choices(Acc_discrete, weights=p_a_i, k=1)[0]
        A_output[i] = a
        
    return A_output

def FVActionPred(model,Input,Input_Vl,mode,device):
    acc_min, acc_max, acc_interval = -9.5, 4, 0.25
    Acc_discrete = np.arange(acc_min, acc_max + acc_interval, acc_interval).round(2).tolist()
    
    input_tensor = torch.tensor(Input, dtype=torch.float32).to(device)
    input_vl_tensor = torch.tensor(Input_Vl, dtype=torch.float32).to(device)
    p_a = model(input_tensor,input_vl_tensor).to(device)
    p_a = torch.exp(p_a[:,-1,:])
    
    A_output = np.zeros(p_a.shape[0])
    for i in range(A_output.shape[0]):
        p_a_i = p_a[i].tolist() # output of instance i
        if mode == "rs": ## Randomly sample (discrete action)
            a = random.choices(Acc_discrete, weights=p_a_i, k=1)[0]
        A_output[i] = a
        
    return A_output



def LVTrajSynthetic(model,V_l_0,A_l_0,V_f_0,A_f_0,T_traj=5,mode="rs",device=device):
    
    t_intv = 0.05 # time interval
    n_traj = V_l_0.shape[0] # number of trajectories
    l_traj = int(T_traj/0.05)+1 # length of lead vehicle velocity
    
    ## pre-define the kinematic info matrix
    ## includes: d_t, v_t, a_{t-1}
    KM_l = np.zeros((n_traj,l_traj,3))
    KM_l[:,0,1] = copy.deepcopy(V_l_0)
    ## set the motion function
    CA_reverse = np.array([[1,-t_intv,t_intv**2/2],[0,1,-t_intv],[0,0,1]])
    
    ## generate input for first output action
    Input = np.zeros((n_traj,l_traj,5))
    Input[:,0,1] = copy.deepcopy(V_l_0)
    Input[:,0,2] = copy.deepcopy(A_l_0)
    Input[:,0,3] = copy.deepcopy(V_f_0)
    Input[:,0,4] = copy.deepcopy(A_f_0)
    
    ## generate following vehicle's output 
    A_output = LVActionPred(model,Input[:,:1,:],mode=mode,device=device)
    A_output[(KM_l[:,0,1]==0)&(A_output>0)] = 0
    KM_l[:,0,2] = copy.deepcopy(A_output)
    
    for i_t in range(1,l_traj):
        
        ## update KM
        KM_l[:,i_t,:] = (CA_reverse @ KM_l[:,i_t-1,:].T).T
        KM_l[KM_l[:,i_t,1]<0,i_t,1] = 0 ## make sure velocity > 0
        
        ## derive new action
        Input[:,i_t,0] = Input[:,i_t-1,0] - t_intv
        Input[:,i_t,1] = copy.deepcopy(KM_l[:,i_t,1])
        Input[:,i_t,2] = copy.deepcopy(KM_l[:,i_t-1,2])
        Input[:,i_t,3] = copy.deepcopy(V_f_0)
        Input[:,i_t,4] = copy.deepcopy(A_f_0)
        A_output = LVActionPred(model,Input[:,:i_t+1,:],mode=mode,device=device)
        A_output[(KM_l[:,i_t,1]==0)&(A_output>0)] = 0
        KM_l[:,i_t,2] = copy.deepcopy(A_output)
    
    return KM_l

def FVTrajSynthetic(model,V_f_0,A_f_0,D_0,V_l_full,mode="rs",device=device):
    
    t_intv = 0.05 # time interval
    n_traj = V_f_0.shape[0] # number of trajectories
    l_traj = V_l_full.shape[1] # length of lead vehicle velocity
    
    ## set the motion function
    CA_reverse = np.array([[1,-t_intv,t_intv**2/2],[0,1,-t_intv],[0,0,1]])
    # CV = np.array([[1,t_intv],[0,1]])
    
    ## pre-define the kinematic info matrix
    KM_f, KM_l = np.zeros((n_traj,l_traj,3)), np.zeros((n_traj,l_traj,2))
    KM_f[:,0,1], KM_f[:,0,2] = V_f_0, A_f_0
    # KM_f[:,0,1] = V_f_0
    KM_l[:,0,0], KM_l[:,:,1] = D_0, V_l_full
    for i_t in range(1,KM_l.shape[1]):
        KM_l[:,i_t,0] = KM_l[:,i_t-1,0] - t_intv * KM_l[:,i_t-1,1]
    
    ## generate input for first output action
    Input = np.zeros((n_traj,l_traj,5))
    Input[:,0,1] = KM_f[:,0,1]
    Input[:,0,2] = KM_f[:,0,1]-KM_l[:,0,1]
    Input[:,0,3] = KM_l[:,0,0]-KM_f[:,0,0]
    Input[:,0,4] = KM_f[:,0,2]
    
    Input_Vl = np.zeros((n_traj,1,97))
    Input_Vl[:,0,:] = V_l_full[:,:97]
    
    ## generate following vehicle's output 
    # A_output = ActionGen_FV(model,Input[:,:1,:],mode=mode,device=device)
    A_output = FVActionPred(model,Input[:,:1,:],Input_Vl,mode=mode,device=device)
    A_output[(KM_f[:,0,1]==0)&(A_output>0)] = 0
    KM_f[:,0,2] = copy.deepcopy(A_output)
    
    for i_t in range(1,l_traj):
        
        ## update KM
        KM_f[:,i_t,:] = (CA_reverse @ KM_f[:,i_t-1,:].T).T
        KM_f[KM_f[:,i_t,1]<0,i_t,1] = 0 ## make sure velocity > 0
        
        ## derive new action
        Input[:,i_t,0] = Input[:,i_t-1,0] - t_intv
        Input[:,i_t,1] = copy.deepcopy(KM_f[:,i_t,1])
        Input[:,i_t,2] = copy.deepcopy(KM_f[:,i_t,1]-KM_l[:,i_t,1])
        Input[:,i_t,3] = copy.deepcopy(KM_l[:,i_t,0]-KM_f[:,i_t,0])
        Input[:,i_t,4] = copy.deepcopy(KM_f[:,i_t-1,2])

        # A_output = ActionGen_FV(model,Input[:,:i_t+1,:],mode=mode,device=device)
        A_output = FVActionPred(model,Input[:,:i_t+1,:],Input_Vl,mode=mode,device=device)
        A_output[(KM_f[:,i_t,1]==0)&(A_output>0)] = 0
        KM_f[:,i_t,2] = copy.deepcopy(A_output)
        
    return KM_l, KM_f
