import numpy as np
import torchvision.transforms as transforms
import torch

# # # for more details on the Game class
# # # refere to the pseudocode available at https://arxiv.org/src/1911.08265v2/anc/pseudocode.py

class Game():
    def __init__(self, 
                 gym_env = None, discount = 0.95, limit_of_game_play = float("inf"), 
                 observation_dimension = None, action_dimension = None, 
                 rgb_observation = None, action_map = None ):
        """
        Init game
        
        Parameters
        ----------
            gym_env (gym_class): 
            The gym env (game) use for the learning and inference.
            Defaults to None.
            
            discount (float): 
            The discount factor for the calcul of the value
            Defaults to 0.95.
            
            limit_of_game_play (int): 
            Maximum number of game allow per selfplay
            Defaults to float("inf").
            
            observation_dimension (int): 
            The dimension of the observation space.
            Defaults to None.
            
            action_dimension (int): 
            The dimension of the action space .
            Defaults to None.
            
            rgb_observation (bool): 
            Bool value True or False that tell you to use the rgb render as observation
            Defaults to None.
            
            action_map (dict): 
            Dict containing the map between integer and possible mouve of the game
            Defaults to None.
        """        
        self.action_history = []
        self.rewards = []
        self.policies = []
        self.discount = discount
        
        self.root_values = []
        self.child_visits = []
        
        self.env = gym_env
        self.observations = []
        self.done = False
        self.limit_of_game_play = limit_of_game_play
        
        self.action_map = action_map
        self.action_space_size = action_dimension
        
        self.rgb_observation = rgb_observation
        shape = observation_dimension[:-1] if type(observation_dimension) == tuple else None #(24,24)

        if shape != None:
            self.transform_rgb = transforms.Compose([lambda x : x.copy().astype(np.uint8), #make a copy of the array and change type to uint8(allow the next transform to rescale)
                                                    transforms.ToTensor(),       #will permute dimension to the appropiate channel for image and rescale between 0 and 1
                                                    transforms.Resize(shape),  #resize the image
                                                    lambda x : x[None,...] ])     #add an extra dimension at the beginning for batch
        else: 
            self.transform_rgb = None
            
        self.game_prio = 0
        self.mouve_prio = 0

        
    @property
    def game_length(self):
        #return the lenght of the game
        return len(self.action_history)
    
    #actualize pygame frame
    def vision(self):
        return self.env.render()
    
    #capture image frm render
    def render(self):
        return self.transform_rgb(self.env.render())

    #to generalize for as much game as possible , those are all the different
    #case i had to deal with.
    def flatten_state(self, state):
        if isinstance(state,tuple):
            state = torch.tensor([i.tolist() for i in state if isinstance(i,np.ndarray)] , 
                                 dtype=torch.float
                                 ).flatten()[None,...]
        elif isinstance(state,list):
            state = torch.tensor(state , 
                                 dtype=torch.float
                                 ).flatten()[None,...]
        elif isinstance(state,np.ndarray):
            state = torch.tensor(state.tolist() ,
                                 dtype=torch.float
                                 ).flatten()[None,...]
        else:
            try: 
                state =  torch.tensor([float(i) for i in state] ,
                                       dtype=torch.float
                                       ).flatten()[None,...]
            except: 
                state = torch.tensor([float(state)] ,
                                         dtype=torch.float
                                         ).flatten()[None,...]
        return state

    def tuple_test_obs(self,x):
        if isinstance(x,tuple):
            x = x[0]
        return x
    
    @property
    def terminal(self):
        #tell you if the game continue or stop with bool value
        return self.done

    def observation(self,observation_shape=None,
                        iteration=0,
                        feedback=None):
        if iteration == 0: 
            state = self.env.reset()
            if self.rgb_observation:
                try:
                    state =  self.tuple_test_obs(self.render())
                except:
                    state = self.transform_rgb(self.tuple_test_obs(state))
            else:
                state = self.flatten_state(self.tuple_test_obs(state))
        else:
            state = feedback[0]
        return state

    
    def store_search_statistics(self, root):
        # store policy without temperature rescale using mcts root first children
        visit_count = np.array([child.visit_count 
                                for child in root.children.values()],
                                dtype=np.float64)
        if visit_count.sum() >= 3:
            policy = visit_count/visit_count.sum()
        else:
            policy = np.array([root.children[u].prior 
                              for u in list(root.children.keys())],
                              dtype=np.float64)
            policy = self.softmax_stable(policy , temperature = 0)
            
        self.child_visits.append(policy)
        self.root_values.append(root.value())
        
    def softmax_stable(self, tensor , temperature = 1):
        if temperature >= 0.3:
            tensor = tensor**(1/temperature)
        return tensor/tensor.sum()

    
    def policy_step(self, policy = None, action = None , temperature = 0 ):
        # if temperature under the treshhold of 0.3 select 
        # the argmax (biggest probability) of policy index inside action
        # or choice randomly if the probability are all equal

        # if temperature over the treshhold of 0.3 select 
        # the select an action base on policy distribution
        # and make sure the policy sum to 1 (can glitch with big number rounding)
        policy = self.softmax_stable(policy , temperature = temperature)
        if temperature > 0.1 or len(set(policy)) == 1:
            selected_action = np.random.choice(action, p=policy)
        else:
            selected_action = action[np.argmax(policy)]

        # save/record the policy during self_play
        # with open(f'report/softmax_model_policy_printed.txt', "a+") as f:
        #     print(selected_action,policy, file=f)

        # # # return one hot encoded action from the discrete action
        action_onehot_encoded = np.zeros(self.action_space_size)
        action_onehot_encoded[selected_action] = 1

        # # # apply mouve and return variable of the env
        step_output = (self.env.step(self.action_map[selected_action]))

        # # # save game variable to a list to return them 
        #contain [observation, reward, done, info] + [meta_data for som gym env]
        step_val = [i for i in step_output]

        if self.rgb_observation : 
            try: observation = self.render()
            except : observation = self.transform_rgb(step_output[0])
                
        else:
            observation = self.flatten_state(step_output[0])

        # # # save game variable to a list to return them 
        #contain [observation, reward, done, info] + [meta_data for som gym env]
        step_val = [observation]+[i for i in step_output[1:]]
        # print(step_val[1:-1])
        # # # done is the parameter of end game [False or True]
        self.done = step_val[2]
        # # # save game variable to class storage
        self.observations.append(observation)
        self.rewards.append(step_val[1])
        self.policies.append(policy)
        self.action_history.append(action_onehot_encoded)
        return step_val
    
    def close(self):
        return self.env.close()
    
    def reset(self):
        self.env.reset()
        
    def make_image(self, index):
        # # # select observation AKA state at specific index
        return self.observations[index]#.reshape(1, -1)

    def make_target(self, state_index, num_unroll, td_steps):
        
        targets = []
        
        for current_index in range(state_index, state_index + num_unroll):
            
            bootstrap_index = current_index + td_steps
            
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else: value = 0
            
            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i 

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else: last_reward = 0

            if current_index < len(self.root_values):
                targets.append([value, last_reward,self.child_visits[current_index]])
            else: targets.append([0, last_reward, np.zeros(self.action_space_size)]) # absorbing state
            
        return targets

    def make_priority(self, td_steps):
        
        target_value = []
    
        for current_index in range(len(self.root_values)):
            
            bootstrap_index = current_index + td_steps

            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else: value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount**i 
                
            if current_index < len(self.root_values):
                target_value.append(value)
            else: target_value.append(0) # absorbing state

        priority_scale = 1 # determine the size of value, if you attempt do use game with huge reward it will renorm them to a more computable unit
        priority_position = np.abs(np.array(self.root_values) - np.array(target_value))**priority_scale 
        priority_game = np.max(priority_position)
        return priority_position , priority_game
