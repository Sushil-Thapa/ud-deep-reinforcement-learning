import torch

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 192        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1.5e-4         # learning rate of the actor 
LR_CRITIC = 1.5e-4      # learning rate of the critic
WEIGHT_DECAY = 0.0001          # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")