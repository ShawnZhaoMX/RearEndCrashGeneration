import torch
import torch.nn as nn

class LVBehaviorModel(nn.Module):
    def __init__(self, input_size, gru_hidden_size, mlp_hidden_size, output_size):
        super(LVBehaviorModel, self).__init__()
        self.gru = nn.GRU(input_size, gru_hidden_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, output_size),  # For 'output_size' discrete acceleration values
            nn.LogSoftmax(dim=2)  # Ensure output is log-probabilities
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        # gru_out = gru_out[:, -1, :]
        output = self.mlp(gru_out)
        return output


class FVBehaviorModel(nn.Module):
    def __init__(self, input_size, gru_hidden_size, mlp_hidden_size, output_size):
        super(FVBehaviorModel, self).__init__()
        self.gru = nn.GRU(input_size, gru_hidden_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_size+128, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, output_size),  # For 'output_size' discrete acceleration values
            nn.LogSoftmax(dim=2)  # Ensure output is log-probabilities
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(97, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # For 'output_size' discrete acceleration values
        )

    def forward(self, x, x2):
        gru_out, _ = self.gru(x)

        mlp2_out = self.mlp2(x2)
        mlp_input_mlp2 = mlp2_out.repeat(1, gru_out.size(1), 1)
        
        combined_out = torch.cat((gru_out, mlp_input_mlp2), dim=2)
        
        output = self.mlp(combined_out)
        return output