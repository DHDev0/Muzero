import torch
import torch.nn as nn
import math

# # # pseudocode available at https://arxiv.org/src/1911.08265v2/anc/pseudocode.py


# # # https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# # # Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Representation__function(nn.Module):
    def __init__(self, 
                 observation_space_dimensions, 
                 state_dimension, 
                 action_dimension, 
                 hidden_layer_dimensions, 
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension
        # # # add to sequence|first and recursive|,, whatever you need
        self.scale = nn.Tanh()
        layernom = nn.LayerNorm(observation_space_dimensions)
        dropout = nn.Dropout(0.5)  # 0.1, 0.2 , 0.25 , 0.5 parameter (first two more recommended for rl)
        activation= nn.LeakyReLU() #, nn.LeakyReLU(), nn.GELU, nn.ReLU(), nn.ELU
        
        first_layer_sequence = [
                                nn.Linear(observation_space_dimensions, hidden_layer_dimensions),
                                activation
                                ]
        
        recursive_layer_sequence = [
                                    nn.Linear(hidden_layer_dimensions,hidden_layer_dimensions),
                                    activation
                                    ]
        
        sequence = first_layer_sequence + (recursive_layer_sequence*number_of_hidden_layer)
        
        self.sequential_container = nn.Sequential(*tuple(sequence))  # # # combine layers
        self.state_norm = nn.Linear(hidden_layer_dimensions, state_dimension)  # # # last layer

    def forward(self, state):
        x = self.sequential_container(state)
        return scale_to_bound_action(self.state_norm(x),self.action_space) 


# # # https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# # # Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Dynamics__function(nn.Module):
    def __init__(self, 
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions, 
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension
        # # # add to sequence|first and recursive|, whatever you need
        self.scale = nn.Tanh()
        layernom = nn.LayerNorm(state_dimension + action_dimension)
        dropout = nn.Dropout(0.5) # 0.1, 0.2 , 0.25 , 0.5 parameter (first two more recommended for rl)
        activation= nn.LeakyReLU() #, nn.LeakyReLU(), nn.GELU, nn.ReLU() , nn.ELU
        
        first_layer_sequence = [
                                nn.Linear(state_dimension + action_dimension, hidden_layer_dimensions),
                                activation
                                ]
        
        recursive_layer_sequence = [
                                    nn.Linear(hidden_layer_dimensions,hidden_layer_dimensions),
                                    activation
                                    ]

        sequence = first_layer_sequence + (recursive_layer_sequence*number_of_hidden_layer)
        
        self.sequential_container = nn.Sequential(*tuple(sequence))  # # # combine layers
        self.reward = nn.Linear(hidden_layer_dimensions, state_dimension)  # # # last layer
        self.next_state_normalized = nn.Linear(hidden_layer_dimensions, state_dimension)  # # # last layer

    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        x = self.sequential_container(x)
        return self.reward(x), scale_to_bound_action(self.next_state_normalized(x),self.action_space)

# # # https://arxiv.org/pdf/1911.08265.pdf [page: 3 and 4] for the structure
# # # Multilayer perceptron (MLP) for muzero with 1D observation and discrete action
class Prediction_function(nn.Module):
    def __init__(self, 
                 state_dimension, 
                 action_dimension, 
                 observation_space_dimensions,
                 hidden_layer_dimensions, 
                 number_of_hidden_layer):
        super().__init__()
        
        # # # add to sequence|first and recursive|,, whatever you need
        layernom = nn.LayerNorm(state_dimension)
        dropout = nn.Dropout(0.5)
        activation= nn.LeakyReLU() #, nn.LeakyReLU(), nn.GELU, nn.ReLU(), nn.ELU
        
        first_layer_sequence = [
                                nn.Linear(state_dimension, hidden_layer_dimensions),
                                activation
                                ]
        
        recursive_layer_sequence = [
                                    nn.Linear(hidden_layer_dimensions,hidden_layer_dimensions),
                                    activation
                                    ]
        
        sequence = first_layer_sequence + (recursive_layer_sequence*number_of_hidden_layer)
        
        self.sequential_container = nn.Sequential(*tuple(sequence))  # # # combine layers
        self.policy = nn.Linear(hidden_layer_dimensions, action_dimension)  # # # last layer
        self.value = nn.Linear(hidden_layer_dimensions, state_dimension)  # # # last layer

    def forward(self, state_normalized):
        x = self.sequential_container(state_normalized)
        return self.policy(x), self.value(x)
    

# # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
# # # To improve the learning process and bound the activations, 
# # # we also scale the hidden state to the same range as
# # # the action input 
def scale_to_bound_action(x,action_space):
    act = torch.tensor([i for i in range(action_space)])
    newmin , newmax = torch.min(act),torch.max(act)
    oldmin , oldmax  = torch.min(x), torch.max(x)
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    scale = newrange / oldrange
    return (x - oldmin) * scale + newmin


# # # L1 Regularization
# # # Explain at : https://paperswithcode.com/method/l1-regularization
def l1(models, l1_weight_decay= 0.0001):
    l1_parameters = []
    for parameter_1,parameter_2,parameter_3 in zip(models[0].parameters(),models[1].parameters(),models[2].parameters()):
        l1_parameters.extend((parameter_1.view(-1), parameter_2.view(-1), parameter_3.view(-1)))
    return l1_weight_decay * torch.abs(torch.cat(l1_parameters)).sum()


# # # https://arxiv.org/pdf/1911.08265.pdf [page: 4]
# # # L2 Regularization manually
# # # or can be done using weight_decay from ADAM or SGD
# # # Explain at : https://paperswithcode.com/task/l2-regularization
def l2(models, l2_weight_decay= 0.0001):
    l2_parameters = []
    for parameter_1,parameter_2,parameter_3 in zip(models[0].parameters(),models[1].parameters(),models[2].parameters()):
        l2_parameters.extend((parameter_1.view(-1), parameter_2.view(-1), parameter_3.view(-1)))
    return l2_weight_decay * torch.square(torch.cat(l2_parameters)).sum()

def cross_entropy(input, target ):
    # (-target * torch.nn.LogSoftmax(dim=0)(input)).mean()
    # (-target * torch.nn.LogSoftmax(dim=0)(input)).sum()
    # (-torch.sigmoid(target) * torch.nn.LogSoftmax(dim=0)(torch.sigmoid(input))).sum()
    # (-torch.sigmoid(target) * torch.nn.LogSoftmax(dim=0)(torch.sigmoid(input))).mean()
    return (-torch.nn.Softmax(dim=-1)(target) * torch.nn.LogSoftmax(dim=-1)(input)).sum()


