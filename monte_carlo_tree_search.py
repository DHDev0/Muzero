import numpy as np
import torch

# # # refere to the pseudocode available at https://arxiv.org/src/1911.08265v2/anc/pseudocode.py

class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = 0
        self.reward = 0
        self.to_play = -1

    def expanded(self):
        return len(self.children) > 0

    def value(self) -> float:
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count


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

class Monte_carlo_tree_search():
    def __init__(self, pb_c_base=19652, pb_c_init=1.25, discount=0.95, root_dirichlet_alpha=0.25, root_exploration_fraction=0.25):
        """
        Init the monte carlos tree search of muzero
        
        Parameters
        ----------
            pb_c_base (int): This is the base value used in the formula for
            calculating the exploration parameter (known as "Pb") in the MCTS
            algorithm. Pb determines the level of exploration that the algorithm
            should perform at each step, with a higher value resulting in more
            expl- oration and a lower value resulting in more exploitation.
            Defaults to 19652.
            
            pb_c_init (float): This is the initial value of the exploration
            parameter Pb. It determines the level of exp- loration that the
            algorithm should perform at the beginning of the search. Defaults to
            1.25.
            
            discount (float): This is the discount factor used in the MCTS al-
            gorithm. It determines the importance of future rewards relative to
            immediate rewards, with a hi- gher discount factor leading to a
            greater emphasis on long-term rewards. Defaults to 0.95.
            
            root_dirichlet_alpha (float): This is the alpha parameter of the
            Dirichlet distr- ibution used in the MCTS algorithm. The Dirichlet
            distribution is used to sample initial move probab- ilities at the
            root node of the search tree, with the alpha parameter controlling
            the level of explo- ration vs exploitation in the sampling process.
            Defaults to 0.25.
            
            root_exploration_fraction (float): This is the exploration fraction
            used in the MCTS algorithm. It determines the proportion of the
            sear- ch time that should be spent exploring the search tree, with a
            higher value resulting in more explora- tion and a lower value
            resulting in more exploitation. Defaults to 0.25.
        """        

        self.reset(pb_c_base, pb_c_init, discount,
                   root_dirichlet_alpha, root_exploration_fraction)

    def reset(self, pb_c_base=19652, pb_c_init=1.25, discount=0.95, root_dirichlet_alpha=0.25, root_exploration_fraction=0.25):
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.root_dirichlet_alpha = root_dirichlet_alpha
        self.root_exploration_fraction = root_exploration_fraction
        self.node = None
        self.model = None
        self.overall_graph = []

    def generate_root_hidden_state(self, observation):

        self.root = Node(0)
        self.min_max_stats = MinMaxStats()
        self.root.hidden_state = self.model.representation_function_inference(
            observation)


    def set_root_to_play_with_the_play_number(self, observation):
        # Monte Carlo Tree Search (MCTS), the to_play variable represents the player
        # whose turn it is to make a move in the current position being considered. This
        # information is used to determine which player's score to update in the MCTS
        # tree, as well as which player's actions to consider when selecting the next
        # move to explore.

        #This configuration always assume the same player is in play.
        self.root.to_play = observation 


    def generate_policy_and_value(self):
        policy, value = self.model.prediction_function_inference(
            self.root.hidden_state)
        return policy, value


    def expand_the_children_of_the_root_node(self, policy, player_to_play=0):
        for i in range(policy.shape[0]):
            for h in range(policy.shape[-1]):
                
                # if self.num_simulations == 0 :
                #     epsilon = np.random.uniform(low=1e-7, high=2e-7, size=1)[0]
                # else:
                #     epsilon = 0 # eps for policy

                self.root.children[h] = Node(prior=policy[i, h])
                self.root.children[h].to_play = self.root.to_play 


    def back_propagate_initial_children(self, value, observation):
        for bnode in reversed([self.root]):
            bnode.value_sum += value if torch.equal(observation, bnode.to_play) else -value
            bnode.visit_count += 1
            self.min_max_stats.update(bnode.value())
            value = bnode.reward + self.discount * value


    def add_exploration_noise_at_the_root(self, train):
        if self.num_simulations == 0 :
            train = False
            
        if train:
            actions = list(self.root.children.keys())
            noise = np.random.dirichlet(
                [self.root_dirichlet_alpha] * len(actions))
            frac = self.root_exploration_fraction
            for a, n in zip(actions, noise):
                self.root.children[a].prior = self.root.children[a].prior * \
                    (1 - frac) + n * frac


    def initialize_history_node_searchpath_variable(self):
        history = []
        self.node = self.root
        search_path = [self.root]
        return history, search_path


    def ucb_score(self, parent, child):
        pb_c = np.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        prior_score = (np.sqrt(parent.visit_count) *  pb_c * child.prior) / (child.visit_count + 1)
        if child.visit_count > 0:
            value_score = self.min_max_stats.normalize( child.reward + self.discount * child.value())
        else:
            value_score = 0

        return prior_score + value_score + np.random.uniform(low=1e-7, high=2e-7, size=1)[0]


    def select_child(self):
        _, action, child = max((self.ucb_score(self.node, child), action, child)
                               for action, child in self.node.children.items())
        return action, child


    def choice_node_to_expand_using_max_ucb_score(self, history, search_path):
        while self.node.expanded():
            action, self.node = self.select_child()
            history.append(action)
            search_path.append(self.node)
        return search_path[-2]


    def generate_reward_and_hidden_state(self, parent, history):
        reward, hidden_state = self.model.dynamics_function_inference(
            parent.hidden_state, history[-1])
        return reward, hidden_state


    def update_reward_and_hidden_state_for_the_chosen_node(self, reward, hidden_state):
        self.node.reward, self.node.hidden_state = reward, hidden_state


    def generate_policy_and_value_for_the_chosen_node(self, hidden_state):
        policy, value = self.model.prediction_function_inference(hidden_state)
        return policy, value


    def create_new_node_in_the_chosen_node_with_action_and_policy(self, policy, player_to_play=0):

        for i in range(policy.shape[0]):
            for h in range(policy.shape[-1]):
                self.node.children[h] = Node(prior=policy[i, h])
                self.node.children[h].to_play = self.root.to_play 


    def back_propagate_and_update_min_max_bound(self, search_path, value):

        for bnode in reversed(search_path):
            bnode.value_sum += value if torch.equal(self.root.to_play, bnode.to_play) else -value
            bnode.visit_count += 1
            self.min_max_stats.update(bnode.value())
            value = bnode.reward + self.discount * value

    def softmax(self,x):
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

    
    def generate_action_and_policy_base_on_node_count(self):

        action = np.array(list(self.root.children.keys()))
        if self.num_simulations >= 1 :
            policy = np.array([self.root.children[u].visit_count for u in list(self.root.children.keys())], dtype=np.float64)
        else:
            policy = np.array([self.root.children[u].prior for u in list(self.root.children.keys())], dtype=np.float64) 
        return policy, self.root, action

    def run(self, observation=None,  num_simulations=10, model=None, train=True):

        self.model = model
        
        self.num_simulations = num_simulations

        self.generate_root_hidden_state(observation)

        self.set_root_to_play_with_the_play_number(observation)

        policy, value = self.generate_policy_and_value()

        self.expand_the_children_of_the_root_node(policy)

        # self.back_propagate_initial_children(value, observation)

        self.add_exploration_noise_at_the_root(train)

        for _ in range(num_simulations):

            history, search_path = self.initialize_history_node_searchpath_variable()

            parent = self.choice_node_to_expand_using_max_ucb_score(
                history, search_path)

            reward, hidden_state = self.generate_reward_and_hidden_state(
                parent, history)

            self.update_reward_and_hidden_state_for_the_chosen_node(
                reward, hidden_state)

            policy, value = self.generate_policy_and_value_for_the_chosen_node(
                hidden_state)

            self.create_new_node_in_the_chosen_node_with_action_and_policy(
                policy)

            self.back_propagate_and_update_min_max_bound(search_path, value)

        policy, root, action = self.generate_action_and_policy_base_on_node_count()

        return policy, root, action


