from torch.utils.data import Dataset
import torch

# Define the dataset
class Dataset_LV(Dataset):
    def __init__(self, Inputs, Outputs, Weights):
        """
        Inputs and Outputs are lists containing Input and Output
        Input and Output are expected to be in shape [num_trajectories, total_length, input_size]
        where each trajectory is a separate sequence.
        """
        self.Inputs = []
        self.Outputs = []
        self.Weights = []
#         self.Weights_t = []
#         ## define time penalty function
#         weight_t = lambda t: (0.1-t)**(-0.5)
        
        ## if velocity is list
        for trajectory_idx, Input in enumerate(Inputs): # number of the trajectory
            Input = Inputs[trajectory_idx][:97]
            Output = Outputs[trajectory_idx][1:97]
            Output = np.append(Output,Output[-1].reshape(1,-1),axis=0)
            weight = Weights[trajectory_idx]
            
            self.Inputs.append(Input)
            self.Outputs.append(Output)
            self.Weights.append(weight)
        
    def __len__(self):
        return len(self.Inputs)

    def __getitem__(self, index):
        return (self.Inputs[index], self.Outputs[index], self.Weights[index])

class Dataset_FV(Dataset):
    def __init__(self, Inputs, Inputs_Vl, Outputs, Weights):
        """
        Inputs and Outputs are lists containing Input and Output
        Input and Output are expected to be in shape [num_trajectories, total_length, input_size]
        where each trajectory is a separate sequence.
        """
        self.Inputs = []
        self.Inputs_Vl = []
        self.Outputs = []
        self.Weights = []
#         self.Weights_t = []
#         ## define time penalty function
#         weight_t = lambda t: (0.1-t)**(-0.5)
        
        ## if velocity is list
        for trajectory_idx, Input in enumerate(Inputs): # number of the trajectory
            Input = Inputs[trajectory_idx][:97]
            Input_Vl = Inputs_Vl[trajectory_idx][:97]
            Output = Outputs[trajectory_idx][1:97]
            Output = np.append(Output,Output[-1].reshape(1,-1),axis=0)
            weight = Weights[trajectory_idx]
            
            self.Inputs.append(Input)
            self.Inputs_Vl.append(Input_Vl)
            self.Outputs.append(Output)
            self.Weights.append(weight)

        self.Inputs = np.array(self.Inputs)
        self.Inputs_Vl = np.array(self.Inputs_Vl)
        self.Outputs = np.array(self.Outputs)
        self.Weights = np.array(self.Weights)
        
    def __len__(self):
        return len(self.Inputs)

    def __getitem__(self, index):
        return (self.Inputs[index], self.Inputs_Vl[index], self.Outputs[index], self.Weights[index])


import numpy as np
import copy

def soft_encode(targets, range_min, range_max, step):
    """
    Encode continuous targets into distributed probabilities between discrete classes.
    Args:
        targets (ndarray): A 1D array of continuous target values.
        range_min (float): Minimum value of the target range.
        range_max (float): Maximum value of the target range.
        step (float): Step between discrete class values.
    Returns:
        ndarray: A 2D array of shape (len(targets), num_classes) with distributed class probabilities.
    """
    # Calculate the number of classes
    num_classes = int((range_max - range_min) / step + 1)
    # Initialize the encoded output array
    encoded = np.zeros((len(targets), num_classes))

    # Loop through each target value and distribute it across the nearest classes
    for i, target in enumerate(targets):
        # Clamp values to be within the range
        clamped_target = max(min(target, range_max), range_min)
        # Find the indices of the nearest lower and upper classes
        lower_idx = int((clamped_target - range_min) // step)
        upper_idx = lower_idx + 1

        # Calculate the proximity to the upper class
        upper_weight = (clamped_target - (lower_idx * step + range_min)) / step
        lower_weight = 1 - upper_weight

        # Distribute the weights accordingly
        encoded[i, lower_idx] = lower_weight
        if upper_idx < num_classes:  # Ensure the upper index is within bounds
            encoded[i, upper_idx] = upper_weight

    return encoded

def gaussian_soft_encode(targets, range_min, range_max, step, std_dev):
    """
    Encode continuous targets based on normal distribution proximity between discrete classes.
    Args:
        targets (ndarray): A 1D array of continuous target values.
        range_min (float): Minimum value of the target range.
        range_max (float): Maximum value of the target range.
        step (float): Step between discrete class values.
        std_dev (float): Standard deviation for the Gaussian distribution.
    Returns:
        ndarray: A 2D array of shape (len(targets), num_classes) with Gaussian distributed class probabilities.
    """
    # Define the number of classes
    num_classes = int((range_max - range_min) / step + 1)
    # Initialize the encoded output array
    encoded = np.zeros((len(targets), num_classes))
    # Define a range of discrete class values
    class_values = np.arange(range_min, range_max + step, step)

    # Loop through each target value and distribute it based on normal distribution
    for i, target in enumerate(targets):
#         if target>-0.75 or target<0.75:
#             acc_std_dev = 0.03
#         else:
#             acc_std_dev = 0.25
        # Calculate the contributions of this target to each class based on the normal distribution
        contributions = np.exp(-0.5 * ((class_values - target) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
        # Normalize contributions to sum up to 1
        contributions /= contributions.sum()
        encoded[i] = contributions

    return encoded


acc_min, acc_max, acc_interval =  -9.5, 7, 0.25
acc_std_dev = 0.07
def LVTrainingData(df, range_min = acc_min, range_max = acc_max, step = acc_interval, std_dev = acc_std_dev):
    '''
    Transform the dataframe read from csv, to the model's input and output
    '''

    Inputs, Outputs, Weights = [], [], []

    for i in range(1,5001):

        # firstly, extract lead vehicle's trajectory of each ID
        weight_i = df[df["id"]==i].iloc[0, 6]
        v_l_i = df[df["id"]==i].iloc[:-1, 3].to_numpy()
        t_i = df[df["id"]==i].iloc[:-1, 1].to_numpy()
        T_s = t_i[1]-t_i[0] # sample time, typically is 0.05s
        a_l_i = (v_l_i[1:] - v_l_i[:-1])/T_s
        a_l_i = np.append(a_l_i,a_l_i[-1])
        # a_l_i = np.gradient(v_l_i)/T_s

        # set FV initial condition as input as well
        v_f_0 = df[df["id"]==i].iloc[-2, 2]
        a_f_0 = (df[df["id"]==i].iloc[-2, 2] - df[df["id"]==i].iloc[-3, 2])/T_s
        v_f_0 = np.tile(v_f_0, t_i.shape[0])
        a_f_0 = np.tile(a_f_0, t_i.shape[0])

        ## filp the data
        v_l_i = np.flip(v_l_i)
        t_i = np.flip(t_i)
        a_l_i = np.flip(a_l_i)
        t_i = t_i-t_i[0]

        #### build input array
        ## The input has time, lead velocity, delta velocity, distance, following history acc
        ## The input has time, LV velocity, LV acceleration, FV velocity at 0s, FV acceleration at 0s
        # Input = np.column_stack((t_i, v_l_i, a_l_i))
        Input = np.column_stack((t_i, v_l_i, a_l_i, v_f_0, a_f_0))
        # Input[1:,2] = Input[:-1,2] ## action should has 1 timestamp lagging

        #### build output array
        P_a_l_i = gaussian_soft_encode(a_l_i, range_min, range_max, step, std_dev)
        Output = copy.deepcopy(P_a_l_i)

        Inputs.append(Input)
        Outputs.append(Output)
        Weights.append(weight_i)

    return Inputs, Outputs, Weights


acc_min, acc_max, acc_interval =  -9.5, 4, 0.25
acc_std_dev = 0.07
def FVTrainingData(df, range_min = acc_min, range_max = acc_max, step = acc_interval, std_dev = acc_std_dev):
    '''
    Transform the dataframe read from csv, to the model's input and output
    '''

    Inputs, Outputs, Weights = [], [], []
    Inputs_Vl = []

    for i in range(1,5001):

        # firstly, extract lead vehicle's trajectory of each ID
        weight_i = df[df["id"]==i].iloc[0, 6]
        v_l_i = df[df["id"]==i].iloc[:-1, 3].to_numpy()
        v_f_i = df[df["id"]==i].iloc[:-1, 2].to_numpy()
        d_i = df[df["id"]==i].iloc[:-1, 4].to_numpy() - df[df["id"]==i].iloc[-2, 4]
        t_i = df[df["id"]==i].iloc[:-1, 1].to_numpy()
        T_s = t_i[1]-t_i[0] # sample time, typically is 0.05s
        # a_l_i = (v_l_i[1:] - v_l_i[:-1])/T_s
        # a_l_i = np.append(a_l_i,a_l_i[-1])
        a_f_i = (v_f_i[1:] - v_f_i[:-1])/T_s
        a_f_i = np.append(a_f_i,a_f_i[-1])
        # a_l_i = np.gradient(v_l_i)/T_s

        ## filp the data
        v_l_i = np.flip(v_l_i)
        v_f_i = np.flip(v_f_i)
        d_i = np.flip(d_i)
        t_i = np.flip(t_i)
        a_f_i = np.flip(a_f_i)
        t_i = t_i-t_i[0]

        delta_v_i = v_f_i - v_l_i

        #### build input array
        ## The input has time, FV velocity, delta velocity, distance, following history acc
        ## The input has time, LV velocity, LV acceleration, FV velocity at 0s, FV acceleration at 0s
        Input = np.column_stack((t_i, v_f_i, delta_v_i, d_i, a_f_i))
        # Input[1:,2] = Input[:-1,2] ## action should has 1 timestamp lagging

        #### build output array
        P_a_f_i = gaussian_soft_encode(a_f_i, range_min, range_max, step, std_dev)
        Output = copy.deepcopy(P_a_f_i)

        Inputs.append(Input)
        Inputs_Vl.append(v_l_i)
        Outputs.append(Output)
        Weights.append(weight_i)

    return Inputs, Inputs_Vl, Outputs, Weights


def evaluate_LVmodel(model, criterion, test_dataloader, device):
    test_losses = []
    with torch.no_grad():
        for inputs, targets, sample_weights in test_dataloader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            sample_weights = sample_weights.to(device)

            log_probs = model(inputs)
            loss = criterion(log_probs, targets)
            weighted_loss = torch.sum(torch.sum(loss,axis=2) * sample_weights.reshape(-1,1))/torch.sum(sample_weights)
            test_losses.append(weighted_loss.item())
    return torch.mean(torch.tensor(test_losses)), torch.min(torch.tensor(test_losses)), torch.max(torch.tensor(test_losses))

def evaluate_FVmodel(model, criterion, test_dataloader, device):
    test_losses = []
    with torch.no_grad():
        for inputs, inputs_vl, targets, sample_weights in test_dataloader:
            inputs = inputs.float().to(device)
            inputs_vl = inputs_vl.reshape(inputs_vl.shape[0],1,inputs_vl.shape[1])
            inputs_vl = inputs_vl.float().to(device)
            targets = targets.float().to(device)
            sample_weights = sample_weights.to(device)

            log_probs = model(inputs,inputs_vl)
            loss = criterion(log_probs, targets)
            weighted_loss = torch.sum(torch.sum(loss,axis=2) * sample_weights.reshape(-1,1))/torch.sum(sample_weights)
            test_losses.append(weighted_loss.item())
    return torch.mean(torch.tensor(test_losses)), torch.min(torch.tensor(test_losses)), torch.max(torch.tensor(test_losses))