import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import json
import gymnasium as gym
from neural_network_mlp_model import l1, l2, weights_init
from neural_network_mlp_model import Loss_function
import itertools
import copy


class Muzero:

    def __init__(self, 
                 model_structure="mlp_model", 
                 observation_space_dimensions=None, 
                 action_space_dimensions=None,
                 state_space_dimensions=9, 
                 k_hypothetical_steps=10,
                 learning_rate=1e-3, 
                 optimizer = "adam",
                 lr_scheduler = None,
                 loss_type = "general", 
                 device='cpu',
                 num_of_epoch=300, 
                 hidden_layer_dimensions=16,
                 number_of_hidden_layer=1, 
                 load=False, 
                 type_format=torch.float32,
                 use_amp=False, 
                 scaler_on=False,
                 bin_method="uniform_bin", 
                 bin_decomposition_number=10):
        """
        Init muzero model
        
        Parameters
        ----------
            model_structure (str):  
                choice between:   
                'mlp_model': MLP with flatten state.  
                'lstm_model': LSTM with flatten state.  
                "transformer_model': Transformer with flatten state.  
                'vision_model': Resnetv2 + MLP with 2D state.  (need to change gym env render_mode to "rgb_array" or "human")
                'vision_conv_lstm_model': Resnetv2 + LSTM with 2D state. (need to change gym env render_mode to "rgb_array" or "human")  
                Defaults to None.  
            
            observation_space_dimensions (gym:env.observation_space):  
                the observation space return by a gym env.  
                Defaults to None.  
            
            action_space_dimensions (gym:env.action_space):  
                the action space return by a gym env.   
                Defaults to None.  
            
            state_space_dimensions (int):   
                Choose an odd number because the state neeed to be split in an array  
                with 0 as origin and negative left side and positive right side  
                example: [-2 , -1 , 0 , 1 , 2] = 5  
                Defaults to None.  
            
            k_hypothetical_steps (int):  
                choice between 0, 5 and 10.  
                Defaults to None.  
            
            learning_rate (float):  
                choice between 0.1, 0.01, 0.001 and 0.0001.   
                Defaults to 1e-3.  
            
            optimizer (str):  
                choice between "adam" or "sgd".  
                Defaults to "adam".  
            
            lr_scheduler (str):  
                Pytorch scheduler  
                choice between "steplr","cosineannealinglr","cosineannealinglrwarmrestarts","onecyclelr" or None.   
                None : Do not apply any scheduler.  
                "steplr" : Decays the learning rate of each parameter group by gamma.  
                "cosineannealinglr" : Decays the learning rate of each parameter using cosine annealing schedule.  
                "cosineannealinglrwarmrestarts" : Decays the learning rate of each parameter using cosine annealing warm restarts schedule.  
                "onecyclelr" : Decays the learning rate according to the 1cycle learning rate policy.  
                Defaults to None.
                
            loss_type (str):  
                choice between "general" and "game",  
                "general": [ value: cross entropy, policy: cross entropy , reward: cross entropy]  
                "game": [ value: mse, policy: cross entropy , reward: 0 ]  
                Defaults to "general".  
            
            device (str): 
                choice between "cuda" or "cpu".  
                "cuda" : Use GPU for training and inference.  
                "cpu" : Use CPU for training and inference.  
                Defaults to 'cpu'.  
                
            num_of_epoch (int):  
                number of epoch.  
                Defaults to 300.  
                
            hidden_layer_dimensions (int):   
                Defaults to 64.  
            
            number_of_hidden_layer (int):  
                Defaults to 1.  
                
            load (bool):  
                choice between True or False.   
                Defaults to False.  
            
            type_format (torch.dtype):  
                choice a pytorch dtype like:  
                torch.float16,   
                torch.bfloat16,  
                torch.float32,   
                torch.float64.  
                Defaults to torch.float32.  
                
            use_amp (bool):   
                choice between True and False to use mix precision  
                Defaults to True.  
                
            scaler_on (bool):  
                Automatically turn on and off following use_amp parameter.  
                Defaults to None.  
                
            bin_method (str):   
                choice between "linear_bin" and "uniform_bin".   
                "linear_bin"  : sample from bound with linear split  
                "uniform_bin" : sample from bound with uniform split  
                Defaults to "uniform_bin".
                
            bin_decomposition_number (int):   
                int : the number of sampled variable from the distribution of bin_method   
                Defaults to 10.  
        """        

        self.reset(model_structure, observation_space_dimensions, action_space_dimensions,
                   state_space_dimensions, k_hypothetical_steps,
                   learning_rate,optimizer,
                   lr_scheduler,loss_type,device,
                   num_of_epoch, hidden_layer_dimensions,
                   number_of_hidden_layer, load,
                   type_format, use_amp,
                   scaler_on, bin_method,
                   bin_decomposition_number)

    def reset(self, model_structure=None, observation_space_dimensions=None,
              action_space_dimensions=None, state_space_dimensions=None,
              k_hypothetical_steps=None, learning_rate=1e-3,
              optimizer = "adam", lr_scheduler = None,loss_type = "general",
              device='cpu', num_of_epoch=300,
              hidden_layer_dimensions=64, number_of_hidden_layer=1,
              load=False, type_format=torch.float32,
              use_amp=False, scaler_on=False,
              bin_method="uniform_bin", bin_decomposition_number=10):

        # # # the size of the encoded/support for value and reward
        self.state_dimension = state_space_dimensions
        
        # # # number of weight for your recursive layer
        self.hidden_layer_dimension = hidden_layer_dimensions
        
        # # # Recursive layer, number of layer between your init layer and end layer
        self.number_of_hidden_layer = number_of_hidden_layer
        
        # # # K future step to simulate in the forward pass and loss function
        self.k_hypothetical_steps = k_hypothetical_steps
        
        # # # type of loss you want, muzero paper show a "general" and "game" loss 
        # # # https://arxiv.org/pdf/1911.08265.pdf [pahe: 19]
        self.loss_type = loss_type
        
        # # # Learning rate of the optimizer
        self.lr = learning_rate
        
        # # # optimizer
        self.opt = optimizer
        
        # # # lr scheduler
        self.sch = lr_scheduler
    
        # # # total number of epoch that one want to compute
        self.epoch = num_of_epoch
        
        # # # count the number of epoch
        self.count = 0
        
        # # # The device to compute on. (CPU or GPU)
        self.device = device
        
        # # # The tensor type for the all process. Set to bfloat16 for cpu
        if self.device == "cpu" and "float16" in str(type_format):
            self.type_format = torch.bfloat16
        else:
            self.type_format = type_format
            
         # # # Variable to enable mix precision 
        if self.device == "cpu" and use_amp: 
            print("Currently, AutocastCPU only support Bfloat16 as the autocast_cpu_dtype")
        if "float16" in str(self.type_format):
            self.use_amp = True
        elif "float64" in str(self.type_format):
            self.use_amp = False
        else:
            self.use_amp = use_amp
        
        # # # Variable to enable scale of the gradient for small tensor type
        self.scaler_on = True if use_amp else scaler_on
        
        # # # Tag number for your model (can use it to save and reload it)
        self.random_tag = np.random.randint(0, 100000000)
        
        # # # Type of desire model, which will set the type of observation.
        self.model_structure = model_structure  # 'vision_model' , 'mlp_model'
        
        # # # init gradient scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # # # allow or not float16 in model matmul operation
        self.fp16backend = "float16" in str(self.type_format)
        
        # # # Unlock float16 for matmul depending on self.fp16backend value
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = self.fp16backend
        
        #set default dtype 
        if not self.use_amp: 
            torch.set_default_dtype(self.type_format)
        
        # # store all compute loss at the end of each epoch
        self.store_loss = []

        self.bin_method = bin_method
        self.bin_decomposition_number = bin_decomposition_number
        
        if not load:
            # # # vision model will use a resize(apply transform in game.py) 98,98,3 RGB image as observation
            # # # mlp_model will flatten the game observation
            self.observation_dimension = self.model_obs(model_structure,observation_space_dimensions)
            
            self.model_repo()
                
            # # # Init gym action space
            action_space = Gym_space_transform(bin=bin_method, mode=bin_decomposition_number)
            # # # will create a disctonary containing all the combinaison of action as a category
            # # # depending on the split bin for continous box
            # # # for discrete it will create the category of all possible mouve
            # # # for discrete and continous box will create a dict of all combinaison and map it as categorical representation
            action_space.design_observation_space(action_space_dimensions)
            # # # your dictionary ( categorical map )
            self.action_dictionnary = action_space.dictionary
            # # # the dimension of the categorical map
            self.action_dimension = action_space.dict_shape[0]
            
            # # # init model 
            self.representation_function = Representation_function(observation_space_dimensions=self.observation_dimension,
                                                                        state_dimension=self.state_dimension,
                                                                        action_dimension=self.action_dimension,
                                                                        hidden_layer_dimensions=self.hidden_layer_dimension,
                                                                        number_of_hidden_layer=self.number_of_hidden_layer).to(self.device)

            self.dynamics_function = Dynamics_function(state_dimension=self.state_dimension,
                                                            action_dimension=self.action_dimension,
                                                            observation_space_dimensions=self.observation_dimension,
                                                            hidden_layer_dimensions=self.hidden_layer_dimension,
                                                            number_of_hidden_layer=self.number_of_hidden_layer).to(self.device)

            self.prediction_function = Prediction_function(state_dimension=self.state_dimension,
                                                                action_dimension=self.action_dimension,
                                                                observation_space_dimensions=self.observation_dimension,
                                                                hidden_layer_dimensions=self.hidden_layer_dimension,
                                                                number_of_hidden_layer=self.number_of_hidden_layer).to(self.device)
            
            self.initiate_model_weight()
            
            # # # If you are not using mix precision, it will set your tensor type.
            self.model_without_amp()
            # # # If the model is on gpu, set parallele batching.
            self.model_parallel()
            # # # tell the model if you are using RGB observation or game state
            self.is_RGB = "vision" in self.model_structure
    
            # # # init your loss function , optimizer and scheduler
            self.init_criterion_and_optimizer()

            
    def model_repo(self):
        # to add a custom model with equivalent structure
        def global_imports(modulename): 
            model_function = ["Representation_function","Dynamics_function","Prediction_function"]
            for i in model_function:   
                context_module = __import__(modulename,fromlist=[model_function])
                globals()[i] = getattr(context_module, i)
                
        # # # Import the model that you are using for training and inference 
        # # # without previously declaring it. ( modular with equivalent class )
        if self.model_structure == 'mlp_model':
            global_imports("neural_network_mlp_model")
        elif self.model_structure == 'lstm_model':
            global_imports("neural_network_lstm_model")
        elif self.model_structure == 'vision_model':
            global_imports("neural_network_vision_model")
        elif self.model_structure == 'vision_conv_lstm_model':
            global_imports("neural_network_vision_conv_lstm_model")
        elif self.model_structure == 'transformer_model':
            global_imports("neural_network_transformer_decoder_model")
            
            
            
            
    def model_obs(self,model_structure,observation_space_dimensions):
        if "vision" in [model_structure]:
            observation_dimension_per_model = (98, 98, 3)
        else:
            observation_dimension_per_model = self.obs_space(observation_space_dimensions)
        return observation_dimension_per_model
    
            
    def model_without_amp(self):
        if not self.use_amp:
            self.representation_function = self.representation_function.type(self.type_format)
            self.dynamics_function = self.dynamics_function.type(self.type_format)
            self.prediction_function = self.prediction_function.type(self.type_format)
                
    def initiate_model_weight(self):
        # initialize the model weight and bias
        self.representation_function.apply(weights_init)
        self.dynamics_function.apply(weights_init)
        self.prediction_function.apply(weights_init)
                
    def model_parallel(self):
        if torch.cuda.device_count() > 1 and self.device != "cpu":
            self.representation_function = torch.nn.DataParallel(
                self.representation_function)
            self.dynamics_function = torch.nn.DataParallel(
                self.dynamics_function)
            self.prediction_function = torch.nn.DataParallel(
                self.prediction_function)



    def init_criterion_and_optimizer(self):
        # # # https://pytorch.org/docs/stable/nn.html#loss-functions
        # # # if you prefer to use pytorch loss function

        
        # refer to : https://arxiv.org/pdf/1911.08265.pdf [page 19]
        if self.loss_type == "general":
            self.criterion_value = Loss_function(parameter = (self.action_dimension),
                                                prediction = ["softmax_transform","zero_clamp_transform"], 
                                                label = ["no_transform"]
                                                ).cross_entropy
            self.criterion_reward = Loss_function(parameter = (self.action_dimension),
                                                prediction = ["softmax_transform","zero_clamp_transform"],
                                                label = ["no_transform"]
                                                ).cross_entropy
            self.criterion_policy = Loss_function(prediction = ["softmax_transform","zero_clamp_transform"],
                                                label = ["no_transform"]
                                                ).cross_entropy

                                            
        if self.loss_type == "general_kkc":
            self.criterion_value = Loss_function(parameter = (self.action_dimension),
                                                prediction = ["softmax_transform","zero_clamp_transform"], 
                                                label = ["zero_clamp_transform"]
                                                ).kldiv
            self.criterion_reward = Loss_function(parameter = (self.action_dimension),
                                                prediction = ["softmax_transform","zero_clamp_transform"],
                                                label = ["zero_clamp_transform"]
                                                ).kldiv
            self.criterion_policy = Loss_function(prediction = ["softmax_transform","zero_clamp_transform"],
                                                label = ["zero_clamp_transform"]
                                                ).cross_entropy

            
        # refer to : https://arxiv.org/pdf/1911.08265.pdf [page 19]
        if self.loss_type == "game":
            self.criterion_value = Loss_function(parameter = (self.action_dimension),
                                                prediction = ["softmax_transform","zero_clamp_transform"], 
                                                label = ["no_transform"]
                                                ).mse
            self.criterion_reward = Loss_function(parameter = (self.action_dimension),
                                                prediction = ["softmax_transform","zero_clamp_transform"],
                                                label = ["no_transform"]
                                                ).zero_loss
            self.criterion_policy = Loss_function(prediction = ["softmax_transform","zero_clamp_transform"],
                                                label = ["no_transform"]
                                                ).cross_entropy
            
        if self.loss_type == "game_mmc":
            self.criterion_value = Loss_function(parameter = (self.action_dimension),
                                                prediction = ["softmax_transform","zero_clamp_transform"], 
                                                label = ["no_transform"]
                                                ).mse
            self.criterion_reward = Loss_function(parameter = (self.action_dimension),
                                                prediction = ["softmax_transform","zero_clamp_transform"],
                                                label = ["no_transform"]
                                                ).mse
            self.criterion_policy = Loss_function(prediction = ["softmax_transform","zero_clamp_transform"],
                                                label = ["no_transform"]
                                                ).cross_entropy
            

        
        # # # model parameter feed to the optimizer
        # # # you can change "lr" to specify particular lr for different model (delete lr= in optim)
        self.params = [{'params': self.representation_function.parameters(), 'lr': self.lr},
                       {'params': self.dynamics_function.parameters(), 'lr': self.lr},
                       {'params': self.prediction_function.parameters(), 'lr': self.lr}]
        # # # an other way to do it: (will take the lr= of your optimizer and apply it to all the model using the optim.)
        # self.params = list(self.representation_function.parameters()) + \
        #               list(self.dynamics_function.parameters()) + \
        #               list(self.prediction_function.parameters())

        # # # Optimizer
        if self.opt == "adam":
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=0)  # weight_decay=1e-4 (pytorch l2 regularization)
        if self.opt == "sgd":
            self.optimizer = optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=0)   # weight_decay=1e-4 (pytorch l2 regularization)

        # # # Learning rate scheduler 
        self.scheduler_lr = ["steplr","cosineannealinglr","cosineannealinglrwarmrestarts","onecyclelr"]
        if self.sch == self.scheduler_lr[0]:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)
        if self.sch == self.scheduler_lr[1]:
            # # https://arxiv.org/pdf/2104.06294.pdf refer at the muzero_unplug paper
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, int(self.epoch))
        if self.sch == self.scheduler_lr[2]:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, int(self.epoch))
        if self.sch == self.scheduler_lr[3]:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=self.epoch)


    # # # the batch reshape is kept to match the
    # # # muzero pseudo code, instead of using a
    # # # more traditional Dataset class object
    # # # and Dataloader
    def reshape_batch(self, batches):
        X, Y = [], []

        # # # Batch :
        # # # [([observation numpy array], [action onehot encoded numpy array[] , list[value, reward, policy])...]
        batch = batches[0]
        # # observation
        # batch of observation (state)
        X.append(torch.cat(tuple(b[0] for b in batch), dim=0).type(
                 self.type_format).to(self.device))

        # # batch of action
        X.extend(torch.tensor([b[1][i].tolist() for b in batch],
                 dtype=self.type_format, device=self.device) for i in range(len(batch[0][1]))) # need to fix

        # # batch of [value, policy, reward]
        Y.extend(
                [
                torch.tensor([[b[2][i][0]] for b in batch], dtype=self.type_format, device=self.device), 
                torch.tensor(np.array([b[2][i][2] for b in batch]), dtype=self.type_format, device=self.device), # need to fix
                torch.tensor([[b[2][i][1]] for b in batch], dtype=self.type_format, device=self.device)
                ] 
                 for i in range(len(batch[0][2]))
                )
        
        batch_importance_sampling_ratio = torch.tensor(batches[1], dtype=self.type_format, device=self.device)
        
        batch_game_position = batches[2]
        
        return X, Y, batch_importance_sampling_ratio, batch_game_position 



    def obs_space(self, obs):
        def checker(container):
            if type(container) == gym.spaces.Discrete:
                return torch.tensor(1)
            if type(container) == gym.spaces.box.Box:
                return torch.prod(torch.tensor(list(container.shape)))

        if type(obs) in [gym.spaces.tuple.Tuple, tuple]:
            return int(sum(checker(i) for i in obs))
        else:
            return int(checker(obs))



    def one_hot_encode(self, action, counter_part):

        if not torch.is_tensor(action):
            action = torch.tensor(action).type(
                torch.int64).to(device=self.device)

        if not self.is_RGB:
            if len(action.size()) == 2:
                pass

            if len(action.size()) == 0:
                action = action[None, ...]
                action = torch.nn.functional.one_hot(
                    action, num_classes=self.action_dimension).type(self.type_format)

        if self.is_RGB:
            if len(action.size()) == 2:
                action = torch.argmax(action, dim=1, keepdim=False)
            if len(action.size()) == 0:
                action = action[None, ...]
            action_one_hot = torch.ones((1,
                                         1,
                                         counter_part.shape[2],
                                         counter_part.shape[3],)
                                        ).to(self.device).type(self.type_format)
            action = torch.cat([((action_select+1) / self.action_dimension) * action_one_hot.clone()
                               for action_select in action], dim=0).type(self.type_format)
        return action



    def training_mode(self):
        # # # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # # Check if the gradient graph is computable or not
        if not self.representation_function.training or not self.dynamics_function.training or not self.prediction_function.training:
            self.representation_function.train()
            self.dynamics_function.train()
            self.prediction_function.train()


    # # # https://arxiv.org/pdf/1911.08265.pdf [page: 14]
    # # # SCALE TRANSFORM for value and reward prediction
    # # # https://arxiv.org/pdf/1911.08265.pdf [page: 14]
    # # # We then apply a transformation φ to the
    # # # scalar reward and value targets in order
    # # # to obtain equivalent categorical representations.
    def transform_with_support(self, x):
        shaper = self.state_dimension
        support_base = torch.full_like(x, 0)
        new_size = support_base.size()[:-1] + (shaper,)
        support_base = support_base.expand(new_size)

        x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

        support_distribution = shaper // 2
        remainder_support_distribution = int(
            2*((shaper/2) - support_distribution))

        x = torch.clamp(x, -support_distribution,
                        support_distribution + remainder_support_distribution)
        sign = torch.sign(x)

        # # # Generate one weight and support on the discrete array.
        # # # one value is enough to reconstruct all.
        support1 = torch.floor(x)
        weight1 = torch.ceil(x) - x

        support1 = (support1).flatten()[None]+support_distribution+1
        support1 = torch.clamp(support1, - (shaper-1),
                                shaper-1).T.clone().type(torch.int64)
        weight1 = (weight1*sign).flatten()[None].T

        support_base = support_base.clone().type(weight1.dtype).scatter_(1, support1, weight1)
        return support_base



    # # # https://arxiv.org/pdf/1911.08265.pdf [page: 14]
    # # # SCALE TRANSFORM for value and reward prediction
    # # # Apply a transformation φ to the scalar reward and value targets in order
    # # # to obtain equivalent categorical representations.
    def inverse_transform_with_support(self, input):
        shaper = self.state_dimension
        support_distribution = int(shaper // 2)
        remainder_support_distribution = int(
            2*((shaper/2) - support_distribution))

        # # # Compute softmax and sum the output to get a
        # # # combine weight and value to inverse the transform
        soft_input = torch.softmax(input, dim=1)
        support_init = torch.tensor(list(
            range(-support_distribution, support_distribution + remainder_support_distribution)))
        support_reformat = support_init.expand(soft_input.shape).type(
            soft_input.dtype).to(device=soft_input.device)
        y = torch.sum(support_reformat * soft_input, dim=1, keepdim=True)
        y = torch.sign(y) * (((torch.sqrt(1 + 4 * 0.001 *
                                            (torch.abs(y) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1)
        return y



    def fitler_empty_loss_then_rescale_gradient_and_sum_loss(self,target,loss,gradient_scale):
        # zero_tensor = torch.tensor(0.0 )
        # if target.nelement() == 0 or torch.isnan(loss) or loss == 0:
        #     loss = torch.nan_to_num(loss) + zero_tensor
        #     self.mean_div -= 1

        self.mean_div += 1
        # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
        # # # divide the gradient loss by 1 / num of unroll (k)
        # # # for board game.
        loss.register_hook(lambda grad: grad * gradient_scale)
        self.loss_nn += loss
        self.loss.append(loss.data.clone().detach().cpu().mean())

    # # # For explaination on the forward implement by pytorch:
    # # # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    # # # https://stephencowchau.medium.com/pytorch-module-call-vs-forward-c4df3ff304b1
    # # compute the forward pass of the model
    def compute_forward(self, X):
        # # # gradient scaling value
        grad_scale = 0.5

        # # # "X[0] is the initial observation state ( observation/or hidden state )
        # # # initial_state -> embedded_state
        state_normalized = self.representation_function(X[0])
        
        # # # embedded_state -> policy , value
        policy, value = self.prediction_function(state_normalized)

        #save output of forward pass
        Y_pred = [[value, policy, 0]]
        
        for k in range(self.k_hypothetical_steps):

            # # # "X[k + 1] is the action onehot encoded of the batch
            one_hot_encode_action = self.one_hot_encode(
                X[k + 1], state_normalized)
             # # # embedded_state , action -> next_embedded_state(next observation) , reward
            reward, next_state_normalized = self.dynamics_function(
                state_normalized, one_hot_encode_action)
            
            # # # next_embedded_state -> policy , value
            policy, value = self.prediction_function(state_normalized)
            
            # # # We also scale the gradient at the start of the dynamics function by 1/2
            # # # This ensures that the total gradient applied to the dynamics function stays constant.
            # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
            # # # Reference to register_hook()
            # # # https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
            next_state_normalized.register_hook(lambda grad: grad * grad_scale)
            
            # next_embedded_state  become the new embedded_state
            state_normalized = next_state_normalized 
            
            #save output of forward pass
            Y_pred.append([value, policy , reward])
        return Y_pred
    

    def evaluate_loss(self):
        
        self.loss_nn = 0.0
        self.new_priority = []
            
        for k , ( pred , target ) in enumerate(zip(self.Y_pred,self.Y)):
            
            gradient_scale = 1.0 / self.k_hypothetical_steps if k > 0 else 1.0

            # # # [pred_value_k_hypothetical_steps vs value_k_hypothetical_steps]
            predict_value_k_hypothetical_steps, target_value_k_hypothetical_steps = pred[0], target[0]
            
            # # transform target value scalar to array equivalent representation than the predicted value array
            target_value_k_hypothetical_steps = self.transform_with_support(
                target_value_k_hypothetical_steps)
            
            # # if you want to make scalar vs scalar instead of matrix vs matrix
            # # predict_value_k_hypothetical_steps = self.inverse_transform_with_support(predict_value_k_hypothetical_steps)
            
            # # Compute loss with criterion of choice
            loss = self.criterion_value(
                predict_value_k_hypothetical_steps,
                target_value_k_hypothetical_steps)
            
            self.fitler_empty_loss_then_rescale_gradient_and_sum_loss(
                target_value_k_hypothetical_steps, loss, gradient_scale)


            # # # [pred_policy_k_hypothetical_steps vs policy_k_hypothetical_steps]
            predict_policy_k_hypothetical_steps, target_policy_k_hypothetical_steps = pred[1], target[1]
                        
            loss = self.criterion_policy(
                predict_policy_k_hypothetical_steps, 
                target_policy_k_hypothetical_steps)
            
            self.fitler_empty_loss_then_rescale_gradient_and_sum_loss(
                target_policy_k_hypothetical_steps, loss, gradient_scale)
            
            
            # # # [pred_reward_k_hypothetical_steps vs reward_k_hypothetical_steps]
            # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
            if k > 0:
                predict_reward_k_hypothetical_steps, target_reward_k_hypothetical_steps = pred[2], target[2]
                
                # # transform target reward scalar to array equivalent representation than the predicted reward array
                target_reward_k_hypothetical_steps = self.transform_with_support(
                    target_reward_k_hypothetical_steps)
                
                # # if you want to make scalar vs scalar instead of matrix vs matrix
                # # predict_reward_k_hypothetical_steps = self.inverse_transform_with_support(predict_reward_k_hypothetical_steps, version="V1")
                                                    
                loss = self.criterion_reward(
                    predict_reward_k_hypothetical_steps, 
                    target_reward_k_hypothetical_steps)
                
                self.fitler_empty_loss_then_rescale_gradient_and_sum_loss(
                    target_reward_k_hypothetical_steps, loss, gradient_scale)
            
            #compute priority to actualize the replay buffer with new value
            priority_scale = 1
            self.new_priority.append(
                (torch.abs(torch.nan_to_num(self.inverse_transform_with_support(pred[0])) - torch.nan_to_num(target[0])
                          )**priority_scale).detach().cpu().to(torch.float32).numpy()
                )
                           
        # # # show backporpagation stack error of the gradient graph if it occur
        # torch.autograd.set_detect_anomaly(True)
        
        # # # L1 regularization
        # self.loss_nn += l1((self.representation_function,
        #                     self.dynamics_function,
        #                     self.prediction_function),
        #                     l1_weight_decay = 0.0001)

        # # L2 regularization
        self.loss_nn += l2((self.representation_function,
                            self.dynamics_function,
                            self.prediction_function),
                            l2_weight_decay = 0.0001)

        if self.batch_importance_sampling_ratio.nelement() != 1:
            self.loss_nn *= self.batch_importance_sampling_ratio
        
        #should be : self.loss_nn = self.loss_nn.sum()  but mean() work beter
        self.loss_nn = self.loss_nn.mean()

            

        
    def backpropagation(self):
        
        # # # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        self.optimizer.zero_grad()
        
        # # use if you want to make gradient cliping betwen unscale and scale
        # if self.device != "cpu" or not self.use_amp : 
        #     self.scaler.unscale_(self.optimizer)
        
        # # # more details at : https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        # # # to implement with mix precision : https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        # # gradient cliping
        # torch.nn.utils.clip_grad_norm_(self.representation_function.parameters(), 1)
        # torch.nn.utils.clip_grad_norm_(self.dynamics_function.parameters(), 1)
        # torch.nn.utils.clip_grad_norm_(self.prediction_function.parameters(), 1)

        # # # more details at : https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
        # # comput backward pass of the gradient graph (backpropagation)
        self.scaler.scale(self.loss_nn).backward() \
            if self.device != "cpu" and self.use_amp \
            else self.loss_nn.backward() 
        
        # # # Performs a single optimization step (optimizer parameter update).
        if self.device != "cpu" and self.use_amp :
            self.scaler.step(self.optimizer) 
        else: self.optimizer.step()
        
        if self.device != "cpu" and self.use_amp : 
            self.scaler.update()

        # # # # # update step in scheduler
        if self.sch in self.scheduler_lr : 
            self.scheduler.step(epoch=self.count)
        #     # # (verbose) print learning rate of the scheduler lr
        #     if self.count % 1 == 0:
        #         print("LEARNING RATE: ",self.scheduler.get_last_lr())

        # # # for custom lr step scheduler
        # # # loss scheduler use in muzero (equivalent to cosine annealing)
        # for g in self.optimizer.param_groups:
        #     new_lr = self.lr * (0.5 * (1 + torch.cos(torch.tensor(np.pi) * self.count / self.epoch)))
        #     g['lr'] = new_lr
        # # (verbose) print learning custom lr every 1 epoch
        # if self.count % 1 == 0:
        #     print("LEARNING RATE: ",new_lr)

        # # count the number of epoch without having to input the epoch value
        self.count += 1
        # # list to store and combine all the computed loss for later analyse
        self.store_loss.append(
            [self.loss_nn.data.clone().detach().cpu()] + list(self.loss))
        
    def train(self, batch):
        
        self.training_mode()
        
        # # list to store the computed loss
        self.loss = []
        self.mean_div = 0
        # # reformate sample_batch() to pytorch batch without dataloader
        self.X, self.Y, self.batch_importance_sampling_ratio, self.batch_game_position  = self.reshape_batch(batch)
        
        if self.use_amp:
            with torch.autocast(device_type=self.device, dtype=self.type_format, enabled=self.use_amp),torch.set_grad_enabled(True):
                self.Y_pred = self.compute_forward(self.X)
                self.evaluate_loss()
        else:
            self.Y_pred = self.compute_forward(self.X)
            self.evaluate_loss()
            
        self.backpropagation()
        
        return self.new_priority , self.batch_game_position
        


    # TODO: accelerate inference : https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
    # https://pytorch.org/TensorRT/getting_started/installation.html#installation
    
    def tensor_test(self,data):
        # # test for input type as pytorch tensor
        if not torch.is_tensor(data):
            data = torch.from_numpy(data.astype(np.float32)).type(
                self.type_format).to(self.device)
        # # test for input tensor device ( should be the same than the model )
        if data.device.type != self.device or data.dtype != self.device:
            data = data.type(self.type_format).to(device=self.device)
        return data
        
    def representation_function_inference(self, state):
        # # set model to eval mode if it is in train mode. (Gradrient graph unable)
        if self.representation_function.training:
            self.representation_function.eval()
        
        if self.use_amp:
            # # compute forward pass without gradient graph
            with torch.autocast(device_type=self.device, dtype=self.type_format,enabled=self.use_amp),torch.no_grad():
                # check/fix for inconsistency in input
                state = self.tensor_test(state)
                # forward pass
                state_normalized = self.representation_function(state)
        else:
            state = self.tensor_test(state)
            state_normalized = self.representation_function(state)
            
        # transfer to cpu
        return state_normalized.detach().cpu()


    def dynamics_function_inference(self, state_normalized, action):
        # # set model to eval mode if it is in train mode. (Gradrient graph unable)
        if self.dynamics_function.training:
            self.dynamics_function.eval()            
        if self.use_amp:
            with torch.autocast(device_type=self.device, dtype=self.type_format,enabled=self.use_amp),torch.no_grad():
                # check/fix for inconsistency in input
                state_normalized = self.tensor_test(state_normalized)
                
                # # # action one_hot encoding to 2D ("2D") or 4D tensor ("4D")
                one_hot_encode_action = self.one_hot_encode(
                    action, state_normalized)
                # forward pass
                reward, next_state_normalized = self.dynamics_function(
                    state_normalized, one_hot_encode_action)
                # transfer next_state to cpu
                next_state_normalized = next_state_normalized.detach().cpu()
                # # # transform reward array to scalar and transfer to cpu
                reward = self.inverse_transform_with_support(
                    reward.type(torch.float)
                    ).detach().flatten().type(torch.float).cpu().numpy()[0]
        else:
            state_normalized = self.tensor_test(state_normalized)
            one_hot_encode_action = self.one_hot_encode(
                action, state_normalized)
            reward, next_state_normalized = self.dynamics_function(
                state_normalized, one_hot_encode_action)
            next_state_normalized = next_state_normalized.detach().cpu()

            reward = self.inverse_transform_with_support(
                reward.type(torch.float)
                ).detach().flatten().type(torch.float).cpu().numpy()[0]
        return reward, next_state_normalized


    def prediction_function_inference(self, state_normalized,reanalyze = False):
        # # set model to eval mode if it is in train mode. (Gradrient graph unable)
        if self.prediction_function.training:
            self.prediction_function.eval()
        if self.use_amp:
            with torch.autocast(device_type=self.device, dtype=self.type_format,enabled=self.use_amp), torch.no_grad():
                # check/fix for inconsistency in input
                state_normalized = self.tensor_test(state_normalized)
                # forward pass
                policy, value = self.prediction_function(state_normalized)
            if self.fp16backend : policy = policy.to(torch.float32) #can't change type inside autocast
        else:
            state_normalized = self.tensor_test(state_normalized)
            policy, value = self.prediction_function(state_normalized)
        # # # softmax the policy output and transfer to cpu
        policy = torch.nn.Softmax(dim=-1)(policy).detach().cpu().numpy()
        # # # transform value array to scalar and transfer to cpu
        if reanalyze:
            value =  self.inverse_transform_with_support(value).detach().flatten().type(torch.float).cpu().numpy()
        else:
            value =  self.inverse_transform_with_support(value).detach().flatten().type(torch.float).cpu().numpy()[0]
        
        return policy,value

    
    def save_model(self, directory="model_checkpoint", tag=None, model_update_condition = None):
        if model_update_condition is True:
            if not os.path.exists(directory):
                os.makedirs(directory)

            if tag != 0:
                self.random_tag = tag

            torch.save(self.representation_function,
                    f'{directory}/{self.random_tag}_muzero_representation_function.pt')
            torch.save(self.dynamics_function,
                    f'{directory}/{self.random_tag}_muzero_dynamics_function.pt')
            torch.save(self.prediction_function,
                    f'{directory}/{self.random_tag}_muzero_prediction_function.pt')

            init_variable = {"observation_space_dimensions": self.observation_dimension,
                            "action_space_dimensions": self.action_dimension,
                            "state_space_dimensions": self.state_dimension,
                            "hidden_layer_dimensions": self.hidden_layer_dimension,
                            "number_of_hidden_layer": self.number_of_hidden_layer,
                            "k_hypothetical_steps": self.k_hypothetical_steps,
                            "learning_rate": self.lr,
                            "optimizer" : self.opt,
                            "loss_type" : self.loss_type,
                            "lr_scheduler" : self.sch,
                            "num_of_epoch": self.epoch,
                            "device": self.device,
                            "random_tag": self.random_tag,
                            "action_map": self.action_dictionnary,
                            "model_structure": self.model_structure,
                            "use_amp": self.use_amp}

            with open(f"{directory}/{self.random_tag}_muzero_init_variables.json", "w") as f:
                json.dump(init_variable, f)


    def load_model(self, model_directory="model_checkpoint", tag=0, observation_space_dimensions=None, type_format=torch.float32, device=None):

        try:
            with open(f"{model_directory}/{tag}_muzero_init_variables.json", 'r') as openfile:
                init_var = json.load(openfile)
        except:
            raise Exception(f"file not found at : {model_directory}/{tag}_muzero_init_variables.json")

        self.reset(observation_space_dimensions=init_var["observation_space_dimensions"],
                   action_space_dimensions=init_var["action_space_dimensions"],
                   state_space_dimensions=init_var["state_space_dimensions"],
                   k_hypothetical_steps=init_var["k_hypothetical_steps"],
                   optimizer = init_var["optimizer"],
                   lr_scheduler = init_var["lr_scheduler"],
                   learning_rate=init_var["learning_rate"],
                   loss_type=init_var["loss_type"],
                   device=device if device != None else init_var["device"],
                   num_of_epoch=init_var["num_of_epoch"],
                   hidden_layer_dimensions=init_var["hidden_layer_dimensions"],
                   number_of_hidden_layer=init_var["number_of_hidden_layer"],
                   load=True,
                   type_format=type_format,
                   use_amp=init_var["use_amp"],
                   model_structure=init_var["model_structure"])
        
        self.observation_dimension = init_var["observation_space_dimensions"]
        
        self.model_repo()
        
        self.action_dictionnary = init_var["action_map"]
        self.action_dimension = torch.tensor(self.action_dictionnary).size(0)
        
        self.representation_function = torch.load(
            f'{model_directory}/{init_var["random_tag"]}_muzero_representation_function.pt').to(self.device)
        self.dynamics_function = torch.load(
            f'{model_directory}/{init_var["random_tag"]}_muzero_dynamics_function.pt').to(self.device)
        self.prediction_function = torch.load(
            f'{model_directory}/{init_var["random_tag"]}_muzero_prediction_function.pt').to(self.device)
        
        self.model_without_amp()
                
        self.model_parallel()
            
        self.random_tag = tag if tag > 0 else init_var["random_tag"]
        self.is_RGB = self.model_structure == 'vision_model'
        self.init_criterion_and_optimizer()


  

class Gym_space_transform:
    def __init__(self, bin=10, mode="uniform_bin"):
        self.bin = bin
        self.mode = mode  # "linear_bin" or "uniform_bin"
        self.dictionary = None
        self.dict_shape = None

    def discrete_to_tensor(self, discrete_container):
        return [torch.arange(discrete_container.n, dtype=torch.int).tolist()]

    def continous_to_tensor(self, box_container):
        
        val_low = torch.tensor(box_container.low)[
            None, ...].T.type(torch.float)
        val_high = torch.tensor(box_container.high)[
            None, ...].T.type(torch.float)
        overall = torch.cat([val_low, val_high], dim=-1)
        
        box_space = []
        if self.mode == "uniform_bin" and "float" in str(box_container.dtype):
            box_space.extend(torch.distributions.uniform.Uniform(
                box_minmax[0], box_minmax[1]).sample([self.bin]).tolist() for box_minmax in overall)

        if self.mode == "linear_bin" and "float" in str(box_container.dtype):
            box_space.extend(torch.linspace(
                box_minmax[0], box_minmax[1], steps=self.bin).tolist() for box_minmax in overall)

        if "int" in str(box_container.dtype):
            box_space.extend(torch.arange(
                box_minmax[0], box_minmax[1], dtype=torch.int).tolist() for box_minmax in overall)

        return tuple(box_space)

    def select_container(self, container):
        if type(container) == gym.spaces.Discrete:
            space = self.discrete_to_tensor(container)
        if type(container) == gym.spaces.box.Box:
            space = self.continous_to_tensor(container)
        return space

    def all_permutation(self, bag):
        return list(itertools.product(*bag))

    def design_observation_space(self, container):
        space_part = []
        if type(container) in [gym.spaces.tuple.Tuple, tuple]:
            for space_container in container:
                space_part.extend((self.select_container(space_container)))
        else:
            space_part.extend((self.select_container(container)))

        if len(space_part) > 1:
            ensemble_of_all_the_permutation = self.all_permutation(space_part)
        else:
            ensemble_of_all_the_permutation = space_part[0]

        self.dictionary = ensemble_of_all_the_permutation
        self.dict_shape = torch.tensor(self.dictionary).size()
