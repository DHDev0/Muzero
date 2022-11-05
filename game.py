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
        #NEED TO EXPLAIN EACH VARIABLE
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
            self.transform_rgb = transforms.Compose([lambda x: x.copy().astype(np.uint8), #make a copy of the array and change type to uint8(allow the next transform to rescale)
                                                    transforms.ToTensor(),       #will permute dimension to the appropiate channel for image and rescale between 0 and 1
                                                    lambda x: x.permute(0,1,2) , #if you need to permute dime
                                                    transforms.Resize(shape),  #resize the image
                                                    lambda x: x[None,...] ])     #add an extra dimension at the beginning for batch
        else: 
            self.transform_rgb = None
        
    def observation(self,observation_shape=None,
                        iteration=0,
                        feedback=None):
        if self.rgb_observation:
            if iteration == 0: 
                self.reset()
                state =  self.render()
            else:
                state = feedback[0]
        else: 
            if iteration == 0:
                state = self.flatten_state(self.env.reset())
            else:
                state = feedback[0]
        return state
    
    def close(self):
        return self.env.close()
    
    def reset(self):
        self.env.reset()
        
    def vision(self):
        return self.env.render()
        
    def render(self):
        try: return self.transform_rgb(self.env.render(mode="rgb_array"))
        except: return "Render not implemented. Try with < mlp_model > instead. "

    def flatten_state(self, state):
        try: return torch.tensor([float(i) for i in state] , dtype=torch.float).flatten()[None,...]
        except: return torch.tensor([float(state)] , dtype=torch.float)[None,...]
    
    @property
    def terminal(self):
        #tell you if the game continue or stop with bool value
        return self.done
    @property
    def game_length(self):
        #return the lenght of the game
        return len(self.action_history)
    
    def store_search_statistics(self, root):
        #store policy without temperature rescale using mcts root first children
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (index for index in range(self.action_space_size))
        self.child_visits.append([
                                  root.children[a].visit_count / sum_visits 
                                  if a in root.children 
                                  else 0
                                  for a in action_space
                                ])
        self.root_values.append(root.value())
    
    def policy_step(self, policy = None, action = None , temperature = 0 ):
        # if temperature under the treshhold of 0.0035 select 
        # the argmax (biggest probability) of policy index inside action
        # or choice randomly if the probability are all equal
        if temperature < 0.0035:
            if len(set(policy)) == 1:
                np.random.choice(action, p=policy)
                selected_action =np.random.choice(action, p=policy)
            else:
                policy = np.argmax(policy)
                selected_action = action[policy]
        
        
        # if temperature over the treshhold of 0.0035 select 
        # the select an action base on policy distribution
        # and make sure the policy sum to 1 (can glitch with big number rounding)
        else:
            policy_dist = policy**(1/temperature)
            policy = policy_dist/policy_dist.sum()
            policy = np.nan_to_num(policy)
            if policy.sum() != 1:
                policy =np.nan_to_num( np.nan_to_num(np.abs(policy)) + np.nan_to_num(np.abs((1-policy.sum())/len(policy)))  )         
            selected_action = np.random.choice(action, p=policy)
        
        # save/record the policy during self_play
        with open(f'result/softmax_model_policy_printed.txt', "a+") as f:
            print(selected_action,policy, file=f)

        # # # return one hot encoded action from the discrete action
        action_onehot_encoded = np.zeros(self.action_space_size)
        action_onehot_encoded[selected_action] = 1

        # # # apply mouve and return variable of the env
        observation, reward, done, info = self.env.step(self.action_map[selected_action])
        
        if self.rgb_observation : 
            observation = self.render()
        else:
            observation = self.flatten_state(observation)
            
        # print("observation shape: ",observation.size() )
        # # # done is the parameter of end game [False or True]
        self.done = done
        
        # # # save game variable to a list to return them 
        step_val = [observation, reward, done, info]
        # print("OBS: ",step_val[0])

        # # # save game variable to class storage
        self.observations.append(observation)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.action_history.append(action_onehot_encoded)
        return step_val

    def make_image(self, index):
        # # # select observation AKA state at specific index
        return self.observations[index]#.reshape(1, -1)

    #NEED TO EXPLAIN EACH STEP
    def make_target(self, state_index, num_unroll, td_steps):
        targets = []
        for current_index in range(state_index, state_index + num_unroll + 1):
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

