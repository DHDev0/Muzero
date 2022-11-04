import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import json
import gym
from neural_network_mlp_model import l1,l2,cross_entropy,scale_to_bound_action  
import itertools

class Muzero:

    def __init__(self, model_structure = None,observation_space_dimensions = None, action_space_dimensions = None,
                       state_space_dimensions = None, k_hypothetical_steps = None, 
                       learning_rate = 1e-3, device='cpu', 
                       num_of_epoch = 300, hidden_layer_dimensions = 64, 
                       number_of_hidden_layer= 1, load = False,type_format = torch.float32,
                       use_amp = True, scaler_on = True, 
                       bin_method = "uniform_bin", bin_decomposition_number = 10 ):
        

        self.reset(model_structure,observation_space_dimensions, action_space_dimensions,
                   state_space_dimensions , k_hypothetical_steps, 
                   learning_rate , device, 
                   num_of_epoch , hidden_layer_dimensions, 
                   number_of_hidden_layer , load,
                   type_format , use_amp,
                   scaler_on , bin_method,
                   bin_decomposition_number )
    
    def reset(self, model_structure = None ,observation_space_dimensions = None, 
                    action_space_dimensions = None, state_space_dimensions = None,
                    k_hypothetical_steps = None , learning_rate = 1e-3, 
                    device = 'cpu', num_of_epoch = 300, 
                    hidden_layer_dimensions = 64, number_of_hidden_layer = 1,
                    load = False, type_format = torch.float32,
                    use_amp = True, scaler_on = True, 
                    bin_method = "uniform_bin", bin_decomposition_number = 10 ):
        
        self.state_dimension = state_space_dimensions
        
        self.hidden_layer_dimension = hidden_layer_dimensions
        self.number_of_hidden_layer = number_of_hidden_layer
        
        self.k_hypothetical_steps = k_hypothetical_steps
        self.lr = learning_rate
        # # # total number of epoch that one want to compute
        self.epoch = num_of_epoch
        # # # count the number of epoch
        self.count = 0 
        
        self.device = device
        self.type_format = type_format
        torch.set_default_dtype(self.type_format)
        self.use_amp = use_amp
        self.scaler_on = scaler_on
        
        self.random_tag = np.random.randint(0,100000000)
        self.model_structure = model_structure # 'vision_model' , 'mlp_model'

        if self.scaler_on:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
        
        if not load:   
            self.observation_dimension = {
                                          'vision_model':(98,98,3) ,
                                          'mlp_model' : self.obs_space(observation_space_dimensions)
                                         }[model_structure]
            
            if self.model_structure == 'vision_model':
                from neural_network_vision_model import Representation__function,Dynamics__function,Prediction_function
            if self.model_structure == 'mlp_model':
                from neural_network_mlp_model import Representation__function,Dynamics__function,Prediction_function    
                        
            action_space = Gym_action_space_transform(bin = bin_method,mode = bin_decomposition_number)
            action_space.design_observation_space(action_space_dimensions)
            self.action_dictionnary = action_space.dictionary 
            self.action_dimension = action_space.dict_shape[0]
                                                                                 
            self.representation_function = Representation__function(observation_space_dimensions=self.observation_dimension, 
                                                                    state_dimension=self.state_dimension, 
                                                                    action_dimension = self.action_dimension,
                                                                    hidden_layer_dimensions=self.hidden_layer_dimension,
                                                                    number_of_hidden_layer=self.number_of_hidden_layer).type(self.type_format).to(self.device)
            
            self.dynamics_function = Dynamics__function(state_dimension=self.state_dimension, 
                                                        action_dimension=self.action_dimension, 
                                                        observation_space_dimensions=self.observation_dimension,
                                                        hidden_layer_dimensions=self.hidden_layer_dimension,
                                                        number_of_hidden_layer=self.number_of_hidden_layer).type(self.type_format).to(self.device)
            
            self.prediction_function = Prediction_function(state_dimension=self.state_dimension, 
                                                           action_dimension=self.action_dimension,
                                                           observation_space_dimensions=self.observation_dimension,
                                                           hidden_layer_dimensions=self.hidden_layer_dimension,
                                                           number_of_hidden_layer=self.number_of_hidden_layer).type(self.type_format).to(self.device)
            
            
            self.is_RGB = "sequential_downsampler" in self.representation_function._modules.keys()
            self.init_criterion_and_optimizer()
        

        
    def init_criterion_and_optimizer(self):
        # # # https://pytorch.org/docs/stable/nn.html#loss-functions
        # # # a bunch of different loss function
        # # # | torch.nn.MSELoss(size_average=None, reduce=None, reduction='sum')  |  
        # # # | cross_entropy | nn.BCEWithLogitsLoss(reduce = "sum") |   
        
        self.criterion_0 = cross_entropy
        self.criterion_1 = cross_entropy
        
        # # # model parameter feed to the optimizer
        # # # you can change "lr" to specify particular lr for different model (delete lr= in optim)
        self.params = [{'params': self.representation_function.parameters(), 'lr': self.lr},
                       {'params': self.dynamics_function.parameters(), 'lr': self.lr},
                       {'params': self.prediction_function.parameters(), 'lr': self.lr}]
        # # # an other way to do it: (will take the lr= of your optimizer and apply it to all model using the optim.)
        #self.params = list(self.representation_function.parameters()) + \
                        # list(self.dynamics_function.parameters()) + \
                        # list(self.prediction_function.parameters())

        # # # Optimizer
        self.optimizer = optim.Adam(self.params, lr=self.lr, 
                                    weight_decay=1e-4)  # weight_decay=1e-4 (pytorch l2 regularization)
        # self.optimizer = optim.SGD(self.params, lr=self.lr,
    #                                momentum=0.9, weight_decay=1e-4)   # weight_decay=1e-4 (pytorch l2 regularization)
        
        # # # Learning rate scheduler ( need to unquote self.scheduler and mouve self.optimizer.step() in train() )
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, 
                                                    # gamma=0.99)
        # # # https://arxiv.org/pdf/2104.06294.pdf refer at the muzero_unplug paper
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,int(self.epoch) )
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,int(epoc))
        
        # # store all compute loss at the end of train
        self.store_loss = []
        
        

    # # # the batch reshape is kept to match the 
    # # # muzero pseudo code, instead of using a 
    # # # more traditional Dataset class object 
    # # # and Dataloader
    def reshape_batch(self, batch):
        X, Y = [], []

        # # # Batch :
        # # # [([observation numpy array], [action onehot encoded numpy array[] , list[value, reward, policy])...]

        # # observation     
        # 1D state batch obs
        if not self.is_RGB:
            X.append( torch.cat( tuple(b[0] for b in batch) , dim=0).type(self.type_format).to(self.device))
        # 4D state RGB batch obs
        if self.is_RGB:
            X.append(torch.cat(tuple(b[0] for b in batch),dim=0).type(self.type_format).to(self.device))

        # # action
        X.extend(torch.tensor([b[1][i].tolist() for b in batch], dtype=self.type_format, device=self.device) for i in range(len(batch[0][1])))

        # # value, reward, policy         
        for i in range(len(batch[0][2])):
            Y.extend((torch.tensor([[b[2][i][0]] for b in batch], dtype=self.type_format, device=self.device), 
                      torch.tensor([[b[2][i][1]] for b in batch], dtype=self.type_format, device=self.device), 
                      torch.tensor([b[2][i][2] for b in batch], dtype=self.type_format, device=self.device)))

        # # delete the first reward 
        # # (no initial reward)
        Y.pop(1)
        return X, Y



    def obs_space(self, obs):
        def checker(container):
            if type(container) == gym.spaces.Discrete:
                return torch.tensor(1)
            if type(container) == gym.spaces.box.Box:
                return torch.prod(torch.tensor(list(container.shape)))

        if type(obs) in [gym.spaces.tuple.Tuple,tuple]:
            return int(sum([checker(i) for i in obs]))
        else:
            return int(checker(obs))
        
        
        
    # # need to reimplement code
    def one_hot_encode(self, action,counter_part):
        
        if not torch.is_tensor(action):
            action = torch.tensor(action).type(torch.int64).to(device=self.device)
        
        if not self.is_RGB:
            if len(action.size()) == 2:
                pass
                
            if len(action.size()) == 0:
                action = action[None,...]
                action = torch.nn.functional.one_hot(action, num_classes = self.action_dimension).type(torch.int)

        if self.is_RGB:
            if len(action.size()) == 2:
                action = torch.argmax(action, dim=1,keepdim=False)
            if len(action.size()) == 0:
                action = action[None,...]
            action_one_hot = torch.ones((1,
                                         1,
                                         counter_part.shape[2],
                                         counter_part.shape[3],)
                                       ).to(self.device).type(self.type_format)
            action = torch.cat([((action_select+1)/ self.action_dimension) * action_one_hot.clone()  for action_select in action],dim=0)
        return action


    # # # https://arxiv.org/pdf/1911.08265.pdf [page: 14]
    # # # SCALE TRANSFORM for value and reward prediction
    # # # https://arxiv.org/pdf/1911.08265.pdf [page: 14]
    # # # We then apply a transformation φ to the
    # # # scalar reward and value targets in order 
    # # # to obtain equivalent categorical representations.
    def transform_with_support(self,x,version = "V1"):
        shaper = self.state_dimension
        support_base = torch.full_like(x,0)
        new_size = support_base.size()[:-1] + (shaper,)
        support_base = support_base.expand(new_size)
        
        x = torch.sign(x)*(((torch.abs(x)+1)**(1/2))-1+(x*0.001))
        
        support_distribution = shaper // 2
        remainder_support_distribution = int(2*((shaper/2) - support_distribution))
        
        x = torch.clamp(x, -support_distribution, support_distribution + remainder_support_distribution)
        sign = torch.sign(x)
        
        # # # first version generate one weight and support on the discrete array.
        # # # one value is enough to reconstruct all
        # # # from modified: https://github.com/werner-duvaud/muzero-general/blob/0c4c335d0492d48f7cb8979d479b2761b5d267fb/models.py#L278

        if version == "V1":
            support1 = torch.floor(x)
            weight1 = torch.ceil(x) - x

            support1 = (support1).flatten()[None]+support_distribution+1
            support1 = torch.clamp(support1, -(shaper-1), shaper-1).T.type(torch.int64)
            weight1 = (weight1*sign).flatten()[None].T
    
            support_base = support_base.clone().scatter_(1, support1, weight1)
            return support_base
        
        # # # second version output the two support and weight on the discrete array.
        # # # the loss will regress on 2 target.
        if version == "V2":
            support1 , support2 = torch.floor(x) , torch.ceil(x)
            weight1 , weight2 = torch.ceil(x) - x, x - torch.floor(x)
            support1 = (support1).flatten()[None,...]+(support_distribution + remainder_support_distribution)
            support2 = (support2).flatten()[None,...]+(support_distribution + remainder_support_distribution)
            weight1 = (weight1*sign).flatten()[None,...]
            weight2 = (weight2*sign).flatten()[None,...]
            
            weight = torch.cat([weight1,weight2]).T
            support = torch.cat([support1,support2]).T.type(torch.int64)
            support = torch.clamp(support, -(shaper-1), shaper-1)
            support_base = support_base.clone().scatter_(1, support, weight)
            return support_base.type(self.type_format).to(device=self.device)



    # # # https://arxiv.org/pdf/1911.08265.pdf [page: 14]
    # # # SCALE TRANSFORM for value and reward prediction
    # # # Apply a transformation φ to the scalar reward and value targets in order 
    # # # to obtain equivalent categorical representations.
    def inverse_transform_with_support(self,input,version = "V1"):
        shaper = self.state_dimension
        support_distribution = int(shaper // 2)
        remainder_support_distribution = int(2*((shaper/2) - support_distribution))
        
        # # # first version compute softmax and sum the output to get a 
        # # # combine weight and value to inverse the transform
        if version == "V1":
            soft_input = torch.softmax(input, dim=1)
            support_init = torch.tensor(list(range(-support_distribution, support_distribution + remainder_support_distribution)))
            support_reformat = support_init.expand(soft_input.shape).type(soft_input.dtype).to(device=soft_input.device)
            y = torch.sum(support_reformat * soft_input, dim=1, keepdim=True)
            y = torch.sign(y) * (((torch.sqrt(1 + 4 * 0.001 * (torch.abs(y) + 1 + 0.001)) - 1) / (2 * 0.001))** 2 - 1)
            return y
        
        # # # Second version without softmax find the 2 biggest weight and their support 
        # # # without softmax and then combine weight and support to inverse the transform.
        if version == "V2":
            soft_input = torch.abs(input)
            y = torch.topk(soft_input,2, dim=-1)

            support = y.indices 
            weight = y.values 
            support = support - (support_distribution + remainder_support_distribution)
            
            y = (weight * support).sum(-1)
            y = torch.sign(y) * (((torch.sqrt(1 + 4 * 0.001 * (torch.abs(y) + 1 + 0.001)) - 1) / (2 * 0.001))** 2 - 1)
            return y[None].T.type(self.type_format).to(device=self.device)
    
    
    
    # # # Quote: "masking the loss if the policy label is all 0.""
    # # # https://www.furidamu.org/blog/2020/12/22/muzero-intuition/
    def mask_zero_absorbing_val(self, original_prediction, original_labels):
        # # extract usefull loss by masking zero value and excluding non zero val
        # # create a 0 tensor on the same device than the input
        mask_filter = torch.tensor([0]).to(original_prediction.device)
        # # create a boolean mask whith False value on dimension -2 ffor row full of zero
        mask = torch.where((original_labels.mean(-1) != mask_filter))
        # # apply mask on input
        masked_prediction = original_prediction[mask]
        # # apply mask on value
        masked_label = original_labels[mask]
        #compute loss with assign criterion
        return masked_prediction, masked_label
    
    
    
    def compute_forward(self, X):
        # # # store all prediction (forward pass) of [value, reward, policy]
        Y_pred = []
        self.test = False
        test= self.test
        # # # "X[0] is the state ( observation/or hidden state )
        state_normalized = self.representation_function(X[0])
        if test: print("state 0 forward: ", X[0][:1] , state_normalized[:1],X[0].size() , state_normalized.size())
        policy, value = self.prediction_function(state_normalized)
        if test: print("POLICY 0 forward: ", policy[:1] , value[:1],policy.size() , value.size())
        Y_pred += [value, policy] 

        for k in range(self.k_hypothetical_steps):
            
            # # # "X[k + 1] is the action onehot encoded of the batch
            one_hot_encode_action = self.one_hot_encode( X[k + 1] ,state_normalized)
            reward, next_state_normalized = self.dynamics_function(state_normalized, one_hot_encode_action)
            
            policy, value = self.prediction_function(state_normalized)
            if test: print("POLICY VALUE K forward: ", policy[:1] , value[:1],policy.size() , value.size()) 
            if test: print("REWARD STATE K forward: ", reward[:1] , next_state_normalized[:1],reward.size() , next_state_normalized.size()) 

            # # # We also scale the gradient at the start of the dynamics function by 1/2
            # # # This ensures that the total gradient applied to the dynamics function stays constant.
            # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
            # # # Reference to register_hook()
            # # # https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
            next_state_normalized.register_hook(lambda grad: grad * 0.5)

            state_normalized = next_state_normalized # # or next hidden state
            Y_pred += [value, reward, policy]
        return Y_pred

    
    
    def train(self, batch):
        
        # # list to store the computed loss
        self.loss = []

        # # # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        # # reset gradient of the optimizer
        self.optimizer.zero_grad(set_to_none=True)

        # # reformate sample_batch() to pytorch batch without dataloader
        X, Y = self.reshape_batch(batch)

        # # # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        # # Check if the gradient graph is computable or not
        # if not self.representation_function.training or not self.dynamics_function.training or not self.prediction_function.training:
        self.representation_function.train(), self.dynamics_function.train(), self.prediction_function.train()

        # # # For explaination on the forward implement by pytorch:
        # # # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        # # # https://stephencowchau.medium.com/pytorch-module-call-vs-forward-c4df3ff304b1
        # # compute the forward pass of the model
        Y_pred = self.compute_forward(X)


        # # # difficulty to unify model lower than float32 without autocast 
        # # # a lot of model function are not float16 compatible
        # # # autocast with mix precision and specific tensor type
        # # # the with need to contain all the loss part
        # if self.device == "cpu" : autocast = torch.cpu.amp.autocast
        # if "cuda" in self.device: autocast = torch.cuda.amp.autocast
        # with autocast(self.use_amp,self.type_format):

        # # [pred_value vs target value]
        predict_value , target_value = self.mask_zero_absorbing_val( Y_pred[0], Y[0] )
        loss = self.criterion_0( predict_value, self.transform_with_support(target_value ,
                                                                            version = "V1") )
        # # init loss incrementing https://arxiv.org/pdf/1911.08265.pdf [page : 4]
        self.loss_nn = loss
        # # save loss for cpu without gradient graph
        self.loss.append(loss.data.clone().detach().cpu())

        # # # [ pred_policy vs policy]
        predict_policy , target_policy = self.mask_zero_absorbing_val( Y_pred[1], Y[1] )
        loss = self.criterion_1( predict_policy, target_policy )
        self.loss_nn += loss
        self.loss.append(loss.data.clone().detach().cpu())

        for k in range(self.k_hypothetical_steps): 

            # # # [pred_value_k_hypothetical_steps vs value_k_hypothetical_steps] 
            predict_value_k_hypothetical_steps , target_value_k_hypothetical_steps =  self.mask_zero_absorbing_val(Y_pred[3*k + 2],Y[3*k + 2] )
            # # transform target value scalar to array equivalent representation than the predicted value array
            loss = self.criterion_0(predict_value_k_hypothetical_steps, 
                                    self.transform_with_support(target_value_k_hypothetical_steps ,
                                                                version = "V1")
                                    )
            # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
            # # # divide the gradient loss by 1 / num of unroll (k)
            # # # for board game. 
            loss.register_hook(lambda grad: grad / self.k_hypothetical_steps )
            self.loss_nn += loss
            self.loss.append(loss.data.clone().detach().cpu())

            # # # [pred_reward_k_hypothetical_steps vs reward_k_hypothetical_steps]
            # # # https://arxiv.org/pdf/1911.08265.pdf [page: 15]
            # # # quote/delete this part to omit the prediction loss for atari game
            predict_reward_k_hypothetical_steps , target_reward_k_hypothetical_steps = self.mask_zero_absorbing_val(Y_pred[3*k + 3],Y[3*k + 3])
            # # transform target reward scalar to array equivalent representation than the predicted reward array
            loss = self.criterion_0( predict_reward_k_hypothetical_steps, 
                                    self.transform_with_support(target_reward_k_hypothetical_steps ,
                                                                version = "V1")
                                    )
            loss.register_hook(lambda grad: grad / self.k_hypothetical_steps )
            self.loss_nn += loss
            self.loss.append(loss.data.clone().detach().cpu())

            # # # [pred_policy_k_hypothetical_steps vs policy_k_hypothetical_steps]
            predict_policy_k_hypothetical_steps , target_policy_k_hypothetical_steps = self.mask_zero_absorbing_val(Y_pred[3*k + 4],Y[3*k + 4])
            loss = self.criterion_1(predict_policy_k_hypothetical_steps,
                                    target_policy_k_hypothetical_steps
                                    )
            loss.register_hook(lambda grad: grad / self.k_hypothetical_steps )
            self.loss_nn += loss
            self.loss.append(loss.data.clone().detach().cpu())

            # # # show backporpagation stack error of the gradient graph if it occur
            # torch.autograd.set_detect_anomaly(True)

        if self.scaler_on:

            self.scaler.scale(self.loss_nn).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for g in self.optimizer.param_groups:
                loss_update = self.lr*(0.8 ** (self.count / self.epoch))
                g['lr'] = loss_update

        else:
            # # L1 regularization 
            # self.loss_nn += l1((self.representation_function,self.dynamics_function,self.prediction_function))

            # # L2 regularization 
            # self.loss_nn += l2((self.representation_function,self.dynamics_function,self.prediction_function))

            # # # more details at : https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # # comput backward pass of the dradient graph
            self.loss_nn.backward()

            # # # more details at : https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
            # # gradient cliping
            # torch.nn.utils.clip_grad_norm_(self.representation_function.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.dynamics_function.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.prediction_function.parameters(), 1)

            # # # Performs a single optimization step (optimizer parameter update).
            self.optimizer.step()

            # # # update step in scheduler
            # self.scheduler.step()
            # # (verbose) print learning rate of the scheduler lr
            # if self.count % 1 == 0:
            #     print("LEARNING RATE: ",self.scheduler.get_last_lr())

            # # # for custom lr step scheduler
            for g in self.optimizer.param_groups:
                loss_update = self.lr*(0.8 ** (self.count / self.epoch))
                g['lr'] = loss_update
            # # (verbose) print learning custom lr every 1 epoch
            # if self.count % 1 == 0:
            #     print("LEARNING RATE: ",loss_update)

        # # count the number of epoch without having to input the epoch value
        self.count+=1
        # # list to store and combine all the computed loss for later analyse
        self.store_loss.append([self.loss_nn.data.detach().cpu()] + list(self.loss))
    
    

    def representation_function_inference(self, state):
            # # set model to eval mode if it is in train mode.
            if self.representation_function.training:
                self.representation_function.eval()
            # # compute forward pass without gradient graph
            with torch.no_grad():
                # # test for input type as pytorch tensor
                if not torch.is_tensor(state):
                    state = torch.from_numpy(state.astype(np.float32)).type(self.type_format).to(self.device)
                # # test for input tensor device ( should be the same than the model )
                if state.device.type != self.device:     
                    state = state.type(self.type_format).to(device=self.device)
                    
                state_normalized = self.representation_function(state)
                return state_normalized.detach().cpu()
 
 
 
    def dynamics_function_inference(self, state_normalized, action): 
        if self.dynamics_function.training:
            self.dynamics_function.eval()
        with torch.no_grad():
            if not torch.is_tensor(state_normalized):
                state_normalized = torch.from_numpy(state_normalized.astype(np.float32)).type(self.type_format).to(device=self.device)
            if state_normalized.device.type != self.device:     
                state_normalized = state_normalized.to(device=self.device)
            
            # # # one_hot encoding to 2D ("2D") or 4D tensor ("4D")
            one_hot_encode_action = self.one_hot_encode(action , state_normalized)
            reward, next_state_normalized = self.dynamics_function(state_normalized,one_hot_encode_action)
            # # # transform reward array to scalar
            return self.inverse_transform_with_support(reward , version = "V1").flatten().detach().cpu().numpy()[0], next_state_normalized.detach().cpu()
    
    
    
    def prediction_function_inference(self, state_normalized):
        if self.prediction_function.training:
            self.prediction_function.eval()
        with torch.no_grad():
            if not torch.is_tensor(state_normalized):
                state_normalized = torch.from_numpy(state_normalized.astype(np.float32)).type(self.type_format).to(device=self.device)
            if state_normalized.device.type != self.device:     
                state_normalized = state_normalized.to(device=self.device)
                
            policy, value = self.prediction_function(state_normalized)
            # # # transform value array to scalar
            return policy.detach().cpu().numpy(), self.inverse_transform_with_support(value , version = "V1").flatten().detach().cpu().numpy()[0]
    
    
    
    def save_model(self,directory = "result", tag = None):
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if tag != 0:
            self.random_tag = tag

        torch.save(self.representation_function, f'{directory}/{self.random_tag}_muzero_representation_function.pt')
        torch.save(self.dynamics_function, f'{directory}/{self.random_tag}_muzero_dynamics_function.pt')
        torch.save(self.prediction_function, f'{directory}/{self.random_tag}_muzero_prediction_function.pt')
        
        init_variable = {"observation_space_dimensions": self.observation_dimension,
                         "action_space_dimensions": self.action_dimension,
                         "state_space_dimensions": self.state_dimension,
                         "hidden_layer_dimensions": self.hidden_layer_dimension,
                         "number_of_hidden_layer": self.number_of_hidden_layer,
                         "k_hypothetical_steps": self.k_hypothetical_steps,
                         "learning_rate": self.lr,
                         "num_of_epoch": self.epoch,
                         "device": self.device,
                         "random_tag": self.random_tag,
                         "action_map": self.action_dictionnary,
                         "model_structure": self.model_structure}
        
        with open(f"{directory}/{self.random_tag}_muzero_init_variables.json", "w") as f:
            json.dump(init_variable, f)
        
        
        
    def load_model(self, model_directory = "result", tag = 0, observation_space_dimensions = None, type_format = torch.float32, device = None ):
        
        with open(f"{model_directory}/{tag}_muzero_init_variables.json", 'r') as openfile:
            init_var = json.load(openfile)
            
        self.reset(observation_space_dimensions = init_var["observation_space_dimensions"], 
                   action_space_dimensions = init_var["action_space_dimensions"],
                   state_space_dimensions = init_var["state_space_dimensions"], 
                   k_hypothetical_steps = init_var["k_hypothetical_steps"], 
                   learning_rate = init_var["learning_rate"] , 
                   device = device if device != None else init_var["device"], 
                   num_of_epoch = init_var["num_of_epoch"], 
                   hidden_layer_dimensions = init_var["hidden_layer_dimensions"], 
                   number_of_hidden_layer = init_var["number_of_hidden_layer"],
                   load = True,
                   type_format = type_format)
        
        self.observation_dimension = {
                                'vision_model':(98,98,3) ,
                                'mlp_model' : self.obs_space(observation_space_dimensions)
                                }[init_var["model_structure"]]
        self.model_structure = init_var["model_structure"]
        if self.model_structure == 'vision_model':
            from neural_network_vision_model import Representation__function,Dynamics__function,Prediction_function
        if self.model_structure == 'mlp_model':
            from neural_network_mlp_model import Representation__function,Dynamics__function,Prediction_function    
        self.action_dictionnary = init_var["action_map"] 
        self.action_dimension = torch.tensor(self.action_dictionnary).size(0)
        self.representation_function = torch.load(f'{model_directory}/{init_var["random_tag"]}_muzero_representation_function.pt').type(self.type_format).to(self.device)
        self.dynamics_function = torch.load(f'{model_directory}/{init_var["random_tag"]}_muzero_dynamics_function.pt').type(self.type_format).to(self.device)
        self.prediction_function = torch.load(f'{model_directory}/{init_var["random_tag"]}_muzero_prediction_function.pt').type(self.type_format).to(self.device)
        self.is_RGB = "sequential_downsampler" in self.representation_function._modules.keys()
        self.init_criterion_and_optimizer()
       
                    
class Gym_action_space_transform:
    def __init__(self,bin = 10,mode = "uniform_bin" ):
        self.bin = bin
        self.mode = mode # "linear_bin" or "uniform_bin"
        self.dictionary = None
        self.dict_shape = None
    
    def discrete_to_tensor(self, discrete_container):
        return [torch.arange(discrete_container.n , dtype= torch.int).tolist()]

    def continous_to_tensor(self,box_container):
        val_low = torch.tensor(box_container.low)[None,...].T.type(torch.float)
        val_high = torch.tensor(box_container.high)[None,...].T.type(torch.float)
        overall = torch.cat([val_low,val_high],dim=-1)
        box_space = []
        if self.mode == "uniform_bin" and "float" in str(box_container.dtype):
            box_space.extend(torch.distributions.uniform.Uniform(box_minmax[0], box_minmax[1]).sample([self.bin]).tolist() for box_minmax in overall)

        if self.mode == "linear_bin" and "float" in str(box_container.dtype):  
            box_space.extend(torch.linspace(box_minmax[0], box_minmax[1], steps=self.bin).tolist() for box_minmax in overall)

        if "int" in str(box_container.dtype):
            box_space.extend(torch.arange(box_minmax[0], box_minmax[1], dtype=torch.int).tolist() for box_minmax in overall)

        return tuple(box_space)
    
    def select_container(self , container ):
        if type(container) == gym.spaces.Discrete:
            space = self.discrete_to_tensor(container)
        if type(container) == gym.spaces.box.Box:
            space = self.continous_to_tensor(container)
        return space
    
    def all_permutation(self,bag):
        return list(itertools.product(*bag))

    def design_observation_space(self, container):
        space_part = []
        if type(container) in [gym.spaces.tuple.Tuple,tuple]:
            for space_container in container:
                space_part.extend((self.select_container( space_container )))
        else:
            space_part.extend((self.select_container( container )))
            
        if len(space_part) > 1:
            ensemble_of_all_the_permutation = self.all_permutation(space_part)
        else:
            ensemble_of_all_the_permutation = space_part[0]
            
        self.dictionary = ensemble_of_all_the_permutation
        self.dict_shape = torch.tensor(self.dictionary).size()