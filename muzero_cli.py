
import sys
import gym

from monte_carlo_tree_search import *
from game import *
from replay_buffer import *
from muzero_model import *
from self_play import *

if __name__ == "__main__":
    print(sys.argv[:])
    with open(str(sys.argv[-1]), 'r') as openfile:
        config = json.load(openfile)

    if str(sys.argv[-2]) == "train":
        # # # set game environment from gym library
        env = gym.make(config["game"]["env"]) # 77min to solve carpole

        # # # the random seed are set to 0 for reproducibility purpose
        # # # good reference about it at : https://pytorch.org/docs/stable/notes/randomness.html
        np.random.seed(config["random_seed"]["np_random_seed"]) # set the random seed of numpy
        torch.manual_seed(config["random_seed"]["torch_manual_seed"]) # set the random seed of pytorch
        env.seed(config["random_seed"]["env_seed"]) # set the random seed of gym env

        # # # init/set muzero model for training and inference
        muzero = Muzero(model_structure = config["muzero"]["model_structure"], # 'vision_model' : will use rgb as observation , 'mlp_model' : will use game state as observation
                        observation_space_dimensions = env.observation_space, # dimension of the observation 
                        action_space_dimensions = env.action_space, # dimension of the action allow (gym box/discrete)
                        state_space_dimensions = config["muzero"]["state_space_dimensions"], # support size / encoding space
                        hidden_layer_dimensions = config["muzero"]["hidden_layer_dimensions"], # number of weight in the recursive layer of the mlp
                        number_of_hidden_layer = config["muzero"]["number_of_hidden_layer"], # number of recusion layer of hidden layer of the mlp
                        k_hypothetical_steps = config["muzero"]["k_hypothetical_steps"], # number of future step you want to be simulate during train (they are mainly support loss)
                        learning_rate = config["muzero"]["learning_rate"], # learning rate of the optimizer
                        num_of_epoch = config["muzero"]["num_of_epoch"], # number of step during training (the number of step of self play and training can be change)
                        device = config["muzero"]["device"], # device on which you want the comput to be made : "cpu" , "cuda:0" , "cuda:1" , etc
                        type_format = torch.float32, # choice the dtype of the model. look at [https://pytorch.org/docs/1.8.1/amp.html#ops-that-can-autocast-to-float16]
                        load = config["muzero"]["load"], # function for loading a save model
                        use_amp = config["muzero"]["use_amp"], # use mix precision for gpu (not implement yet)
                        scaler_on = config["muzero"]["scaler_on"], # scale gradient to reduce computation
                        bin_method = config["muzero"]["bin_method"], # "linear_bin" , "uniform_bin" : will have a regular incrementation of action or uniform sampling(pick randomly) from the bound
                        bin_decomposition_number = config["muzero"]["bin_decomposition_number"]) # number of action to sample from low/high bound of a gym discret box

        # # # init/set the game storage(stor each game) and dataset(create dataset) generate during training
        replay_buffer = ReplayBuffer(window_size = config["replaybuffer"]["window_size"], # number of game store in the buffer
                                     batch_size = config["replaybuffer"]["batch_size"], # batch size is the number of observe game during train
                                     num_unroll = muzero.k_hypothetical_steps, # number of mouve/play store inside the batched game
                                     td_steps = config["replaybuffer"]["td_steps"], # number of step the value is scale on 
                                     game_sampling = config["replaybuffer"]["game_sampling"], # 'uniform' or "priority" (will game randomly or with a priority distribution)
                                     position_sampling = config["replaybuffer"]["position_sampling"]) # 'uniform' or "priority" (will sample position in game randomly or with a priority distribution)

        # # # init/set the monte carlos tree search parameter
        mcts = Monte_carlo_tree_search(pb_c_base = config["monte_carlo_tree_search"]["pb_c_base"] , 
                                       pb_c_init = config["monte_carlo_tree_search"]["pb_c_init"], 
                                       discount = config["monte_carlo_tree_search"]["discount"], 
                                       root_dirichlet_alpha = config["monte_carlo_tree_search"]["root_dirichlet_alpha"], 
                                       root_exploration_fraction = config["monte_carlo_tree_search"]["root_exploration_fraction"])

        # # # ini/set the Game class which embbed the gym game class function
        gameplay = Game(gym_env = env, 
                        discount = config["gameplay"]["discount"],
                        limit_of_game_play = config["gameplay"]["limit_of_game_play"], # maximum number of mouve
                        observation_dimension = muzero.observation_dimension, 
                        action_dimension = muzero.action_dimension,
                        rgb_observation = muzero.is_RGB,
                        action_map = muzero.action_dictionnary)
        
        # # # train model (if you choice vison model it will render the game)
        epoch_pr , loss , reward = learning_cycle(number_of_iteration = config["learning_cycle"]["number_of_iteration"], # number of epoch(step) in  muzero should be the |total amount of number_of_iteration x number_of_training_before_self_play|
                                                  number_of_self_play_before_training = config["learning_cycle"]["number_of_self_play_before_training"], # number of game played record in the replay buffer before training
                                                  number_of_training_before_self_play = config["learning_cycle"]["number_of_training_before_self_play"], # number of epoch made by the model before selplay
                                                  number_of_mcts_simulation = config["learning_cycle"]["number_of_mcts_simulation"], 
                                                  model_tag_number = config["learning_cycle"]["model_tag_number"], # tag number use to generate checkpoint
                                                  verbose = config["learning_cycle"]["verbose"], # if you want to print the epoch|reward|loss during train
                                                  muzero_model = muzero,
                                                  gameplay = gameplay,
                                                  number_of_monte_carlo_tree_search = mcts,
                                                  replay_buffer = replay_buffer)

    if str(sys.argv[-2]) == "play":
        play_game_from_checkpoint(game_to_play = config["game"]["env"],
                                    
                                  model_tag = config["play_game_from_checkpoint"]["model_tag"],
                                  model_device = config["play_game_from_checkpoint"]["model_device"],
                                  model_type = torch.float32,
                                    
                                  mcts_pb_c_base = config["monte_carlo_tree_search"]["pb_c_base"] , 
                                  mcts_pb_c_init = config["monte_carlo_tree_search"]["pb_c_init"], 
                                  mcts_discount = config["monte_carlo_tree_search"]["discount"], 
                                  mcts_root_dirichlet_alpha = config["monte_carlo_tree_search"]["root_dirichlet_alpha"], 
                                  mcts_root_exploration_fraction = config["monte_carlo_tree_search"]["root_exploration_fraction"],
                                  mcts_with_or_without_dirichlet_noise = config["play_game_from_checkpoint"]["mcts_with_or_without_dirichlet_noise"],
                                  number_of_monte_carlo_tree_search_simulation = config["play_game_from_checkpoint"]["number_of_monte_carlo_tree_search_simulation"],
                                    
                                  gameplay_discount = config["gameplay"]["discount"],
                                    
                                  temperature = config["play_game_from_checkpoint"]["temperature"],
                                  game_iter = config["play_game_from_checkpoint"]["game_iter"],
                                    
                                  slow_mo_in_second = config["play_game_from_checkpoint"]["slow_mo_in_second"],
                                  render = config["play_game_from_checkpoint"]["render"],
                                  verbose = config["play_game_from_checkpoint"]["verbose"])



