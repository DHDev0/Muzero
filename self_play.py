from game import *
from replay_buffer import *
from monte_carlo_tree_search import *
import torchvision.transforms as transforms
import copy

def play_game(environment = None,
              model = None,
              monte_carlo_tree_search = None , 
              number_of_monte_carlo_tree_search_simulation = 50,
              temperature = 1): 

    environment = copy.deepcopy(environment)
    counter = 0
    observation_reward_done_info = None
    while not environment.terminal and counter < environment.limit_of_game_play:
        state = environment.observation(iteration = counter,
                                        feedback = observation_reward_done_info)
        policy,tree,action = monte_carlo_tree_search.run(observation = state, 
                                                         model = model, 
                                                         num_simulations= number_of_monte_carlo_tree_search_simulation,
                                                         train=True)
        observation_reward_done_info = environment.policy_step(policy = policy, 
                                                               action = action,
                                                               temperature =  temperature)
        environment.store_search_statistics(tree)
        counter+=1
    environment.close()
    return environment

def scaler(x,newmin=0,newmax=1):
    # bound a serie between new value
    oldmin , oldmax  = min(x), max(x)
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    if oldrange == 0:        # Deal with the case where rvalue is constant:
        if oldmin < newmin:      # If rvalue < newmin, set all rvalue values to newmin
            newval = newmin
        elif oldmin > newmax:    # If rvalue > newmax, set all rvalue values to newmax
            newval = newmax
        else:                    # If newmin <= rvalue <= newmax, keep rvalue the same
            newval = oldmin
        normal = [newval for _ in x]
    else:
        scale = newrange / oldrange
        normal = [(v - oldmin) * scale + newmin for v in x]
    return np.array(normal)

def temperature_scheduler(epoch,actual_epoch):
    # # # personal add
    # # # will scale the remperature to an opposite tanh distribution ( 1 - tanh )
    # # # of chosen bound ( look like cosineannealing for reference)
    # array = np.array([i for i in range(1,epoch+1)])
    # index = np.where(array == actual_epoch)
    # range_scale_array = np.tanh(scaler(array,newmin=0.001,newmax=0.75))[index]
    # temperature = (1 - range_scale_array) * 1.1
    # return temperature

    # # # https://arxiv.org/pdf/1911.08265.pdf [page: 13]
    # # # original temperature distrubtion of muzero 
    # # # Temperature is find for choicing an action such as:
    # # # policy**1/T/sum(policy**1/T)
    # # # using the policy output by the mcts
    # # # | under 50%  T=1 | under 75% T=0.5 | over 75% T=0.25
    if epoch * 0.5 > actual_epoch:
        return 1.0
    elif epoch * 0.75 > actual_epoch:
        return 0.5
    else:
        return 0.25


def learning_cycle(number_of_iteration = 10000,
                   number_of_self_play_before_training = 1,
                   number_of_training_before_self_play = 1,
                   number_of_mcts_simulation = 11,
                   model_tag_number = 124,
                   verbose = True,
                   muzero_model = None,
                   gameplay = None,
                   number_of_monte_carlo_tree_search = None,
                   replay_buffer = None):  
    
    # # # Training
    reward , cache_reward , epoch_pr , loss , cache_loss  = [] , [] , [] , [] , []
    for ep in range(1,number_of_iteration+1):
        # # # reset the cache reward for every iteration
        cache_reward , cache_loss = [] , []
        # # # self_play
        for self_play in range(number_of_self_play_before_training):
            # # # run game with mcts prediction and run step from the policy output 
            game = play_game(environment = gameplay, 
                             model = muzero_model,
                             monte_carlo_tree_search = number_of_monte_carlo_tree_search, 
                             number_of_monte_carlo_tree_search_simulation = number_of_mcts_simulation,
                             temperature = temperature_scheduler(number_of_iteration+1,ep)) 
            # # # save all the necessary parameter during play_game fortraining
            replay_buffer.save_game(game)
            # # # save the cumulative reward of each self_play
            cache_reward.append(sum(game.rewards))
        # # # sum the average reward of all self_play
        reward.append(sum(cache_reward)/len(cache_reward))
        # # # save best model. self_play serve as dataset and performace test
        if reward[-1] == max(reward):
            muzero_model.save_model(directory = "result", tag= model_tag_number)
        # # # train model from all game accumulate in the buffer (of the replay_buffer)
        for _ in range(number_of_training_before_self_play):
            muzero_model.train(replay_buffer.sample_batch())
            cache_loss.append(muzero_model.store_loss[-1][0])
        loss.append(sum(cache_loss)/len(cache_loss))
        
        prompt_feedback = f'EPOCH {ep} || reward: {reward[-1]} || loss: { loss[-1] }||'
        epoch_pr.append(prompt_feedback)
        
        if verbose:
            print(prompt_feedback)
            
    return epoch_pr , loss , reward



def play_game_from_checkpoint(game_to_play = 'CartPole-v1',
                              model_tag = 124,
                              model_device = "cuda:0",
                              model_type = torch.float32,
                              mcts_pb_c_base = 19652 , 
                              mcts_pb_c_init = 1.25, 
                              mcts_discount = 0.95, 
                              mcts_root_dirichlet_alpha = 0.25, 
                              mcts_root_exploration_fraction = 0.25,
                              mcts_with_or_without_dirichlet_noise = True,
                              number_of_monte_carlo_tree_search_simulation = 11,
                              gameplay_discount = 0.997,
                              temperature = 0,
                              game_iter = 2000,
                              slow_mo_in_second = 0.0,
                              render = True,
                              verbose = True):

    import gym
    import time 
    from monte_carlo_tree_search import Node , MinMaxStats , Monte_carlo_tree_search
    from game import Game
    from muzero_model import Muzero , Gym_action_space_transform

    #play with model of choice (will repeat variable for explanatory purpose)
    # # # choice game env
    env = gym.make(game_to_play)
    # # # initialize model class without initializing a neural network
    muzero = Muzero(load=True, 
                    type_format = model_type)
    # # # load save model with tag number
    muzero.load_model(tag=model_tag,
                    observation_space_dimensions = env.observation_space, 
                    device= model_device) # set device for model compute
    # # # init the mcts class
    monte_carlo_tree_search = Monte_carlo_tree_search(pb_c_base=mcts_pb_c_base , 
                                                        pb_c_init=mcts_pb_c_init, 
                                                        discount= mcts_discount, 
                                                        root_dirichlet_alpha=mcts_root_dirichlet_alpha, 
                                                        root_exploration_fraction=mcts_root_exploration_fraction)
    # # # create the game class with gameplay/record function
    gameplay = Game(env, 
                    discount = gameplay_discount,
                    observation_dimension = muzero.observation_dimension, 
                    action_dimension = muzero.action_dimension,
                    rgb_observation = muzero.is_RGB,
                    action_map = muzero.action_dictionnary)
    # # # slow animation of the render ( in second )
    sleep = slow_mo_in_second
    # # # number of simulation for the monte carlos tree search
    number_of_monte_carlo_tree_search_simulation = number_of_monte_carlo_tree_search_simulation
    # # # temperature set to 0 will use argmax as policy (highest probability action)
    # # # over a temperature of 0.0035 it will sample with the propability associate to the mouve , picking uniformly
    temperature = temperature
    # # # number of iteration (mouve play during the game)
    game_iter = game_iter

    observation_reward_done_info = None
    # # # or while not environment.terminal: # for loop to bypass env terminal limit, else use while loop and add a counter variable incrementing
    for counter in range(game_iter):
        #render the env
        if render:
            gameplay.vision()
        # # #laps time to see a slow motion of the env
        time.sleep(sleep)
        # # # start the game and get game initial observation / game return observation after action
        state = gameplay.observation(iteration = counter,
                                        feedback = observation_reward_done_info)
        # # # run monte carlos tree search inference
        # # Train [False or True] mean with or without dirichlet at the root
        mcts = copy.deepcopy(monte_carlo_tree_search)
        policy,tree,action = mcts.run(observation = state, 
                                      model = muzero, 
                                      num_simulations= number_of_monte_carlo_tree_search_simulation,
                                      train=mcts_with_or_without_dirichlet_noise)
        
        # # # select the best action from policy and inject the action into the game (.step())
        observation_reward_done_info = gameplay.policy_step(policy = policy, 
                                                               action = action,
                                                               temperature =  temperature)
        
        # # # reset mcts class to empty cache variable
        mcts.reset()
        
        # # # print the number of mouve, action and policy
        if verbose:
            print(f"Mouve number: {counter+1} , Action: {muzero.action_dictionnary[action[np.argmax(policy/policy.sum())]]}, Policy: {policy/policy.sum()}")
    gameplay.close()


    