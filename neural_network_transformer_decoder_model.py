import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),)

    def forward(self, x):
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class decoder_only_transformer(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_positions, num_vocab, num_classes):
        super(decoder_only_transformer, self).__init__()

        self.embed_dim = embed_dim
        self.voc = num_vocab
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)
        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        length, batch = x.shape
        # I choice to rescale float array to int but you could use nn.linear with no grad to embedding input or skip the embedding
        h = self.token_embeddings((x*1000).long())
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits.mean(-2)


class Representation_function(nn.Module):
    def __init__(self,
                 observation_space_dimensions,
                 state_dimension,
                 action_dimension,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        self.action_space = action_dimension
        self.state_norm = nn.Linear(observation_space_dimensions, state_dimension)
    def forward(self, state):
        return scale_to_bound_action(self.state_norm(state))


class Dynamics_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        self.action_space = action_dimension
        self.head = 2
        self.vocab = 1001
        self.batchsize = 128
        self.reward = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)
        self.next_state_normalized =  decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)


    def forward(self, state_normalized, action):
        x = torch.cat([state_normalized.T, action.T]).T
        return self.reward(x), scale_to_bound_action(self.next_state_normalized(x))

class Prediction_function(nn.Module):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 observation_space_dimensions,
                 hidden_layer_dimensions,
                 number_of_hidden_layer):
        super().__init__()
        
        self.head = 2
        self.vocab = 1001
        self.batchsize = 128
        print(f"Batch size is set to: {self.batchsize}")
        print(f"Your model must have the same batch size of {self.batchsize} or you have to change the batch size parameter in neural_network_transformer_decoder_model.py")
        self.policy = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, action_dimension)
        self.value = decoder_only_transformer(hidden_layer_dimensions, self.head, number_of_hidden_layer, self.batchsize, self.vocab, state_dimension)

    def forward(self, state_normalized):
        return self.policy(state_normalized), self.value(state_normalized)


def scale_to_bound_action(x):
    min_next_encoded_state = x.min(1, keepdim=True)[0]
    max_next_encoded_state = x.max(1, keepdim=True)[0]
    scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
    scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
    next_encoded_state_normalized = (
    x - min_next_encoded_state
    ) / scale_next_encoded_state
    return next_encoded_state_normalized 

        
        



