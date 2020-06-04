import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HIDDEN_SIZE = 32
HIDDEN_SIZE_2 = 256


class Net(nn.Module):
    def __init__(self, state_size, actions, no_intruders, verbose=False):
        super().__init__()

        self.actions = actions
        self.action_size = len(actions)
        self.state_size = state_size
        self.no_intruders = no_intruders

        self.verbose = verbose

        if self.verbose:
            print("###################\nInitilizing Model\n###################")
        self.def_layers()

    # Create a definition of layers
    def def_layers(self):
        pass
