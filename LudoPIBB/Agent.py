import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NNet(nn.Module):

    def __init__(self,state_size, action_size ):
        super(NNet, self).__init__()

        self.fc1 = nn.Linear(state_size, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, action_size)

    def forward(self, x):
        x1 = torch.sigmoid(self.fc1(x))
        x2 = torch.sigmoid(self.fc2(x1))
        x3 =  self.fc3(x2)
        return x3


class Agent:
    def __init__(self,state_size, action_size ):
        self.nn = NNet(state_size, action_size)
        self.params = None
        self.shape_params = None
        self.len_params = None
        self.get_weights()

    def get_weights(self):
        #messy but works, possibly a way better implementation of all weight 
        # get/set functions as a lot of possible redundant lists are used
        i = 0
        # list for shape of params pr. layer bias and weigth , all the parameters and the number of params pr layer weight and bias seperate
        shapes = []
        params = []
        len_params = []
        
        for  name, param in self.nn.named_parameters():
            # read weight or bias from layer

            #skip bias
            if name[4:] == "bias":
                continue

            param = param.detach().numpy()
            
            #save shape 
            shapes.append(param.shape)
            # save number of parameters
            len_params.append(np.product(param.shape))
            # add to list of all params
            params.append(np.ravel(param))
        # combine to single numpy array and save    
        self.params = np.concatenate(params)
        self.shape_params = shapes
        self.len_params = len_params

        return self.params

    def get_action(self, state):
        #pass state through network
        output = torch.from_numpy(state).float()
        return self.nn(output).detach().numpy()

    def get_n_weights(self):
        print(len(self.params))
        return len(self.params)


    def set_weights(self, weights):
        with torch.no_grad():
            i = 1
            for name, param in self.nn.named_parameters():

                #skip bias
                if name[4:] == "bias":
                    continue

                # set param to the part of the list of total weights
                param_weights = weights[sum(self.len_params[:i-1]):sum(self.len_params[:i])]
                # reshape it to original shape after ravel and copy to the parameter
                param.copy_(torch.from_numpy(np.resize(param_weights, self.shape_params[i-1])))
                i += 1

class temp_agent(Agent):
    pass