import numpy as np
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

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            if self.game_sampling == "priority":
                self.prio_game.pop(0)
            if self.position_sampling == "priority":
                self.prio_position.pop(0)
                
        if "priority" in [self.game_sampling,self.position_sampling]:
            # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
            # # # ν is the search value and z the observed n-step return
            p_i = [abs((value*(0.997**lenght_game))-lenght_game) for value,lenght_game in zip(game.rewards,list(range(1,1+len(game.rewards)))) ]
            
        if self.position_sampling == "priority":
            # # # individual p_i value for each position
            self.prio_position.append(p_i)
            
        if self.game_sampling == "priority":
            # # # average p_i value for each position
            self.prio_game.append(sum(p_i))
            
        # # # save the game into the buffer(storage)
        self.buffer.append(game)

    def sample_game(self):
        # # # # Sample game from buffer either uniformly or according to some priority.
        # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
        
        if self.game_sampling == "priority":
        # # # priority sampling
            soft_prio_game = np.array(self.prio_game) / np.sum(np.array(self.prio_game))
            return np.random.choice(self.buffer, p=soft_prio_game.tolist())
        
        if self.position_sampling == "uniform":
            # uniform sampling
            return np.random.choice(self.buffer)
        
    def sample_position(self, game):
        # # # Sample position from game either uniformly or according to some priority.
        # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
        if game.game_length-1 > 0:
            if self.position_sampling == "priority":
                # # priority sampling
                targ = self.buffer.index(game)
                soft_prio_position = np.array(self.prio_position[targ])/np.sum(np.array(self.prio_position[targ]))
                return np.random.choice(list(range(game.game_length)), p=soft_prio_position.tolist())
            
            if self.position_sampling == "uniform":
                # # uniform sampling
                return np.random.randint(0, game.game_length-1)
        else:
            return 0

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
        games = [self.sample_game() for _ in range(self.batch_size)] 
        # # # contain: [(<GameLib.Game object at 000000000000>, 5).....]
        # # # return a game and position inside this game choice uniformly(random)
        # # # or according to some priority
        game_pos = [(g, self.sample_position(g)) for g in games] 
        # # # return [([state(the observation)], [action array(onehot encoded)], [value, reward, policy]), ... *batch_size]
        # # # They are your X: [[state(the observation)], [action array(onehot encoded)],...] and Y: [[value, reward, policy],...]
        return [(
                g.make_image(i),
                self.fill_gap_empty_action(g.action_history[i:]),
                g.make_target(i, self.num_unroll, self.td_steps)
                ) for (g, i) in game_pos]
        # # # Why did i do this ugly code ?
        # # # i wanted to implement as close as 
        # # # the muzero pseucode for anyone that 
        # # # want to study or replicate it in the future