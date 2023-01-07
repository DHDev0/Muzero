import numpy as np
import torch
# # # https://arxiv.org/pdf/1911.08265.pdf [page: 3, 13]
class ReplayBuffer():
    def __init__(self, 
                 window_size,
                 batch_size, 
                 num_unroll, 
                 td_steps,
                 game_sampling = "uniform",
                 position_sampling = "uniform"
                 ):
        """
        Init replay buffer
        
        Parameters
        ----------
            window_size (int): Maximum number of game store in the replay buffer
            (each self_play add one game and take at one if the replay buffer is
            full)
            
            batch_size (int): Number of game sample in the batch
            
            num_unroll (int): number of mouve for each game in the batch 
            
            td_steps (int): The td_step is a learning step that compares
            expected and observed rewards and transitions in the environment to
            update and improve the prediction model.
            
            game_sampling (str): choice between "uniform" and "priority".
            "uniform": pick a game randomly in the buffer 
            "priority": pick a game according to a priority distribution
            Defaults to"uniform".
            
            position_sampling (str): choice between "uniform" and "priority".
            "uniform": pick a mouve inside a game randomly in the buffer
            "priority": pick a mouve inside a game according to a priority distribution
            ration in the buffer . 
            Defaults to "uniform".
        """        
        

        #NEED TO EXPLAIN EACH VARIABLE
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll = num_unroll
        self.td_steps = td_steps
        self.buffer = []
        # self.device = device
        
        self.game_sampling = game_sampling
        self.position_sampling = position_sampling
        
        self.prio = []
        self.prio_position = []
        self.prio_game = []
        self.big_n_of_importance_sampling_ratio = 0

    def save_game(self, game):
        
        if len(self.buffer) > self.window_size:
            self.big_n_of_importance_sampling_ratio -= self.buffer[0].game_length
            self.buffer.pop(0)
            if self.game_sampling == "priority":
                self.prio_game.pop(0)
            if self.position_sampling == "priority":
                self.prio_position.pop(0)
            
                
        if "priority" in [self.game_sampling,self.position_sampling]:
            # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
            # # # ν is the mcts.root_value (search value) and z the generated target_value with td_step(observed n-step return)
            p_i_position, p_i_game = game.make_priority( self.td_steps )
            
            # # # individual p_i value for each position
            self.prio_position.append(p_i_position)
            
            # # # average p_i value for each game
            self.prio_game.append(p_i_game)
            self.soft_prio_game = np.array(self.prio_game) / np.sum(np.array(self.prio_game))
            
        # # # save the game into the buffer(storage)self.buffer[0].game_length
        self.buffer.append(game)
        self.big_n_of_importance_sampling_ratio += game.game_length

    def sample_game(self):
        # # # # Sample game from buffer either uniformly or according to some priority.
        # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
        
        if self.game_sampling == "priority":
        # # # priority sampling
            position =  np.random.choice(range(self.soft_prio_game.size), p=self.soft_prio_game)
                        
        elif self.game_sampling == "uniform":
        # uniform sampling
            position = np.random.choice(self.buffer)
                        
        return position
    
    def sample_position(self, game):
        tag = self.buffer.index(game)
        soft_prio_position = self.prio_position[tag]/self.prio_position[tag].sum()
        self.buffer[tag].mouve_prio = soft_prio_position
        
        if game.game_length == 0:
            raise Exception("Game need to return at least one reward")
        
        elif self.position_sampling == "priority":
            # # priority sampling
            position =  np.random.choice(list(range(len(soft_prio_position))), p=soft_prio_position)
            

        elif self.position_sampling == "uniform":
            # # uniform sampling
            position =  np.random.randint(0, game.game_length-1)

        
        return position
    
    def fill_gap_empty_action(self, actions):
        # # # Add [0,0] to keep uniform len inside the batch 
        # # # if the num_unroll is too big for the sample
        # # # The zero sequence will be mask later on in the loss
        # # # They are absorbing state
        actions = actions[:self.num_unroll]
        lenght_action_against_num_unroll = (self.num_unroll - len(actions))
        if lenght_action_against_num_unroll > 0:
            actions += [np.zeros(actions[0].shape)] * lenght_action_against_num_unroll
        return actions
    
    def sample_batch(self):
         # # # contain: [<GameLib.Game object at 0x0000000000>,.....]
         # # # return a game choice uniformly(random) or according to some priority
         # [(game,game_index),...]
        games_pos = [(self.buffer[i],i) for i in [self.sample_game() for _ in range(self.batch_size)]]
        # # # contain: [(<GameLib.Game object at 000000000000>, 5).....]
        # # # return a game and position inside this game choice uniformly(random)
        # # # or according to some priority
        # [(game,game_index, game_position_origin_index),etc...]
        game_pos_and_mouve_pos = [(g, g_p, self.sample_position(g)) for g,g_p in games_pos] 
                
        # # # batch : [([state(the observation)], [action array(onehot encoded)], [value, reward, policy]), ... *batch_size]
        # # # They are your X: [[state(the observation)], [action array(onehot encoded)],...] and Y: [[value, reward, policy],...]
        bacth = [(
                g.make_image(m_p),
                self.fill_gap_empty_action(g.action_history[m_p:]),
                g.make_target(m_p, self.num_unroll, self.td_steps)
                ) for (g, g_p, m_p) in game_pos_and_mouve_pos]

        #np.array([[game_index, game_position_origin_index],...])
        game_pos = np.array([(i[1],i[2]) for i in game_pos_and_mouve_pos])
        
        if "priority" in [self.game_sampling,self.position_sampling] :
            #P(i)
            priority = np.array([self.soft_prio_game[i[1]] * self.buffer[i[1]].mouve_prio[ i[2] ] for i in game_pos_and_mouve_pos ])
            # 1/n * 1/P(i)
            importance_sampling_ratio = 1 / ( self.big_n_of_importance_sampling_ratio * priority )
            
            
            return (bacth , importance_sampling_ratio , game_pos)
            # # # Why did i do this ugly code ?
            # # # i wanted to implement as close as 
            # # # the muzero pseucode for anyone that 
            # # # want to study or replicate the paper in the future
        else:
            return (bacth , np.array([0]) , game_pos)
            
        
    def update_value(self,new_value,position):
        for count,i in enumerate(position):
            remainder = 0
            lenght_game = self.buffer[i[0]].game_length - 1
            for h in range(i[1],min(self.num_unroll + i[1] , lenght_game)):
                self.prio_position[i[0]][h] = new_value[remainder][count][0]
                remainder += 1
            # update game priority by using the biggest priority values as the game priority
            self.prio_game[i[0]] = np.max(self.prio_position[i[0]])
            