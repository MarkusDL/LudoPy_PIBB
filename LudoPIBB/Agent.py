import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NNet(nn.Module):

    def __init__(self,state_size, action_size ):
        super(NNet, self).__init__()

        self.fc1 = nn.Linear(state_size, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, action_size)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3

    def set_weights(self, weights):
        pass
        '''
            with torch.no_grad():
            self.fc1.weight = torch.nn.Parameter(K)
            self.fc2.weight = torch.nn.Parameter(K)
            self.fc3.weight = torch.nn.Parameter(K)
        '''



class Agent:
    def __init__(self,state_size, action_size ):
        self.nn = NNet(state_size, action_size)
        self.params = None
        self.shape_params = None
        self.len_params = None
        self.get_weights()

    def get_weights(self):
        i = 0
        shapes = []
        params = []
        len_params = []
        for  name, param in self.nn.named_parameters():
            param = param.detach().numpy()
            shapes.append(param.shape)
            len_params.append(np.product(param.shape))
            params.append(np.ravel(param))
        self.params = np.concatenate(params)
        self.shape_params = shapes
        self.len_params = len_params

        return self.params

    def get_action(self, state):
        input = torch.from_numpy(state).float()
        return torch.argmax(self.nn(input))

    def get_n_weights(self):
        return len(self.params)


    def set_weights(self, weights):
        with torch.no_grad():
            i = 1
            for name, param in self.nn.named_parameters():
                param_weights = weights[sum(self.len_params[:i-1]):sum(self.len_params[:i])]
                param.copy_(torch.from_numpy(np.resize(param_weights, self.shape_params[i-1])))
                i += 1
