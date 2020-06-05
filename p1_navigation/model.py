import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """ Actor Policy (Q Network) model """
    
    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256, fc3_units=64):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            fcx_units (int): Dimension of hidden sizes, x = ith layer
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units) # 37, 512
        self.fc2 = nn.Linear(fc1_units, fc1_units)  # 512,512
        self.fc3 = nn.Linear(fc1_units, fc2_units)  # 512,256
        self.fc4 = nn.Linear(fc2_units, fc3_units)  # 512,256
        self.fc5 = nn.Linear(fc3_units, action_size) # 256, 4
        
        
    def forward(self, state):
        x = self.fc1(state) # 37 -> 512
        x = F.relu(x)
        x = self.fc2(x) # 512 -> 512
        x = F.relu(x)
        x = self.fc3(x)  # 256 -> 256
        x = F.relu(x)
        x = self.fc4(x)  # 256 -> 64
        x = F.relu(x)
        action = self.fc5(x) # 64 -> 4
        return action
