import math
import numpy as np
import torch

# # # https://arxiv.org/pdf/1911.08265.pdf [page: 1,2 and 12]
# # # pseudocode available at https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.to_play = -1
        
    def expanded(self): 
        return len(self.children) > 0

    def value(self) -> float:
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

# # # https://arxiv.org/pdf/1911.08265.pdf [page: 1,2 and 12]
# # # pseudocode available at https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
class MinMaxStats(object):
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

# # # https://arxiv.org/pdf/1911.08265.pdf [page: 1,2 and 12]
# # # pseudocode available at https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
# # # mcts is atomize for explanatory purpose and later attempt at leaf parallelizing
class Monte_carlo_tree_search():   
    def __init__(self,pb_c_base=19652 , pb_c_init=1.25, discount= 0.95, root_dirichlet_alpha=0.25, root_exploration_fraction=0.25):
        self.reset(pb_c_base , pb_c_init, discount, root_dirichlet_alpha, root_exploration_fraction)
        
    def reset(self,pb_c_base=19652 , pb_c_init=1.25, discount= 0.95, root_dirichlet_alpha=0.25, root_exploration_fraction=0.25):
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.node = None
        self.model = None
        self.overall_graph = []
        
    def generate_root_hidden_state(self,observation):
        self.root = Node(0)
        self.min_max_stats =  MinMaxStats()
        self.root.hidden_state = self.model.representation_function_inference(observation)
        
    def set_root_to_play_with_last_observation(self,observation):
        self.root.to_play = observation
        
    def generate_policy_and_value(self): 
        policy, value = self.model.prediction_function_inference(self.root.hidden_state)
        value = value
        policy = policy/policy.sum()
        return policy, value
    
    def expand_the_children_of_the_root_node(self,policy):
        for i in range(policy.shape[0]):
            for h in range(policy.shape[-1]):
                self.root.children[h] = Node(prior=policy[i,h])
                self.root.children[h].to_play = -self.root.to_play
                
    
    def back_propagate_initial_children(self, value, observation): 
        for bnode in reversed([self.root]):
            bnode.value_sum += value if torch.equal(observation,bnode.to_play) else -value
            bnode.visit_count += 1
            # print(self.root.value().shape)
            self.min_max_stats.update(self.root.value())
            value = bnode.reward + self.discount * value
    
    def add_exploration_noise_at_the_root(self,train):
        if train:
            actions = list(self.root.children.keys())
            noise = np.random.dirichlet([self.root_dirichlet_alpha] * len(actions))
            frac = self.root_exploration_fraction
            for a, n in zip(actions, noise):
                self.root.children[a].prior = self.root.children[a].prior * (1 - frac) + n * frac
    
    def initialize_history_node_searchpath_variable(self): 
        history = []
        self.node = self.root
        search_path = [self.root]
        return history, search_path
    
    def ucb_score(self,parent: Node, child: Node) -> float:

        pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        if child.visit_count > 0:
            value_score = self.min_max_stats.normalize(child.reward + self.discount * child.value())
        else:
            value_score = 0

        return prior_score + value_score   
    
    def select_child(self):
        _, action, child = max((self.ucb_score(self.node, child), action, child) for action, child in self.node.children.items())
        return action, child
    
    def choice_node_to_expand_using_max_ucb_score(self,history,search_path): 
        while self.node.expanded():
            action, self.node = self.select_child()
            history.append(action)
            search_path.append(self.node)
        return search_path[-2]
    
    def generate_reward_and_hidden_state(self,parent,history):
        reward, hidden_state = self.model.dynamics_function_inference(parent.hidden_state, history[-1])
        return reward, hidden_state

    def update_reward_and_hidden_state_for_the_chosen_node(self,reward, hidden_state):
        self.node.reward, self.node.hidden_state = reward , hidden_state
        
    def generate_policy_and_value_for_the_chosen_node(self,hidden_state): 
        policy, value = self.model.prediction_function_inference(hidden_state)
        value = value
        policy = policy/policy.sum()
        return policy, value
    
    def create_new_node_in_the_chosen_node_with_action_and_policy(self,policy):
        for i in range(policy.shape[0]):
            for h in range(policy.shape[-1]):
                self.node.children[h] = Node(prior=policy[i,h])
                self.node.children[h].to_play = -self.node.to_play
    
    def back_propagate_and_update_min_max_bound(self,search_path,value):
        for bnode in reversed(search_path):
            bnode.value_sum += value if torch.equal(self.root.to_play,bnode.to_play) else -value
            bnode.visit_count += 1
            self.min_max_stats.update(self.node.value())
            value = bnode.reward + self.discount * value
    
    def overall_expand(self,lister):
        if len(lister) == 0:
            return [] , []
        l_action,l_node  = [],[]
        for nodes in lister:
          key,val = list(nodes.children.keys()) , list(nodes.children.values())
          for a,b in zip(key,val):
            l_action.append(a)
            l_node.append(b)
        return l_action, l_node
    
    #One could just do [a,b for a,b in root.children.items()] to retrieve 
    #action and self.node.count, count is the policy that going to be average.
    # # # look at generate_action_and_policy_base_on_node_count()
    def retrieve(self,lister,memory_action = [[],],memory_repeat = []):
        def tree(nodes,hist_1=[],hist_2=[]):
            saving_pack_of_count = []
            for h in nodes:
                saving_pack_of_count.append(h.visit_count)
            hist_2.append(saving_pack_of_count)
            l_action, l_node = self.overall_expand(nodes)
            if len(l_action)+len(l_node) !=0:
                hist_1.append(l_action)
                tree(l_node,hist_1=hist_1,hist_2=hist_2)
        tree(lister,hist_1=memory_action,hist_2=memory_repeat)
        return memory_action,memory_repeat
    
    def generate_action_and_policy_base_on_node_count(self): 
        # # # if you want to visualize the tree
        # tree = [self.root]
        # memory_action ,memory_repeat = [[],] , []
        # a , b = self.retrieve(tree,memory_action = memory_action,memory_repeat = memory_repeat)
        # # # # mcts graph
        # print("||||||||||||||||||||||||||||||||||||||")
        # for act,count in zip(a,b):
        #     print("ACTION : ",act)
        #     print("COUNT  : ",count)
        # self.overall_graph += [a] + [b]
        # action = np.array(a[1])
        # policy = np.array(b[1])
        
        action = np.array(list(self.root.children.keys()))
        policy = np.array([self.root.children[u].visit_count for u in list(self.root.children.keys())])
        
        return policy, self.root, action
    
    
    def run(self, observation = None,  num_simulations=10, model=None, train=True):
        
        self.model = model
 
        self.generate_root_hidden_state(observation)

        self.set_root_to_play_with_last_observation(observation)

        policy, value = self.generate_policy_and_value()

        self.expand_the_children_of_the_root_node(policy)

        self.back_propagate_initial_children(value,observation)

        self.add_exploration_noise_at_the_root(train)


        #start simulation
        for _ in range(num_simulations):
            
            history , search_path = self.initialize_history_node_searchpath_variable()

            parent = self.choice_node_to_expand_using_max_ucb_score(history,search_path)

            reward, hidden_state = self.generate_reward_and_hidden_state(parent,history)

            self.update_reward_and_hidden_state_for_the_chosen_node(reward, hidden_state)

            policy, value = self.generate_policy_and_value_for_the_chosen_node(hidden_state) 

            self.create_new_node_in_the_chosen_node_with_action_and_policy(policy)

            self.back_propagate_and_update_min_max_bound(search_path,value)

        policy, root, action = self.generate_action_and_policy_base_on_node_count()
    
        return policy, root, action

