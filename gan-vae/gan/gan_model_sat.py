import argparse
import os
import numpy as np
import math
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from laed.utils import INT, FLOAT, LONG, cast_type
from gan.torch_utils import GumbelConnector


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.flush_valid = False
        self.config = config
        self.kl_w = 0.0
        self.gumbel_connector=GumbelConnector(config.use_gpu)

    def cast_gpu(self, var):
        if self.use_gpu:
            return var.cuda()
        else:
            return var

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        if type(inputs)==list:
            return cast_type(Variable(torch.Tensor(inputs)), dtype,
                         self.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)


    def forward(self, *input):
        raise NotImplementedError

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        # total_loss += self.l2_norm()
        total_loss.backward()
        # self.clip_gradient()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0.0
        for key, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss
    
    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

    def get_optimizer(self, config):
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(self.parameters(), lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(self.parameters(), lr=config.init_lr,
                                       momentum=config.momentum)
    def clip_gradient(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)

    def l2_norm(self):
        l2_reg = None
        for W in self.parameters():
            if l2_reg is None:
                l2_reg = W.norm(2)
            else:
                l2_reg = l2_reg + W.norm(2)
        return l2_reg * self.config.l2_lambda


class WoZGenerator(BaseModel):
    def __init__(self, config):
        super(WoZGenerator, self).__init__(config)
        
        state_in_size = config.state_noise_dim
        action_in_size = config.action_noise_dim
        turn_in_size = config.action_noise_dim
        self.state_out_size = config.state_in_size
        self.state_out_size_final = config.state_out_size
        action_out_size = config.action_num
        
        self.gumble_length_index = self.gumble_index_multiwoz()
        self.gumble_num = len(self.gumble_length_index)
        self.last_layers = nn.ModuleList()
        for gumbel_width in self.gumble_length_index:
            self.last_layers.append(nn.Linear(self.state_out_size, gumbel_width))
        

        self.state_model = nn.Sequential(
            # original: 5 block + 1 linear
            nn.Linear(state_in_size, 100),
            nn.ReLU(True),

            nn.Linear(100, 128),
            nn.ReLU(True),

            nn.Linear(128, self.state_out_size),
        )
        # '''
        self.common_model = nn.Sequential(
            nn.Linear(state_in_size, state_in_size),
            nn.ReLU(True),
            
        )
        # '''
        self.action_model_2 = nn.Sequential(
            nn.Linear( self.state_out_size, 96),            
            nn.ReLU(True),
            
            nn.Linear(96, 96),
            nn.ReLU(True),

            nn.Linear(96, action_out_size)
        )
    def gumble_index_multiwoz(self):
        index = 4  * [6] + 3 * [2] + 60 * [2] + 204 * [2] + 3 * [2] + 2 * [3] + 1 * [2] + 4 * [2] + 4 * [3] \
                 + 1 * [2] + 1 * [2] + 1 * [3] + 1 * [2] + 4 * [2] + 7 * [3] + 1 * [2] + 1 * [2] + 4 * [3] + 1 * [2] \
                 + 3 * [2] + 5 * [3] + 1 * [2] + 1 * [2] + 1 * [2] + 1 * [5]
        return index
    def gumble_index_multiwoz_binary(self):
        index = 9 * [2]
        return index

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()
        self.clip_gradient()

    def forward(self, s_z):
        # state and action share one common MLP
        state_action_turn_pair = self.common_model(self.cast_gpu(s_z))
        # state_rep = torch.tanh(self.state_model(state_action_turn_pair))
        state_rep1 = self.state_model(state_action_turn_pair)
        input_to_gumble = []
        for layer, g_width in zip(self.last_layers, self.gumble_length_index):
            out = layer(state_rep1)
            out = self.gumbel_connector.forward_ST_soft(out.view(-1,  g_width), self.config.gumbel_temp)
            input_to_gumble.append(out)
        state_rep = torch.cat(input_to_gumble, -1)
        # print(state_rep.size())
        state_rep = state_rep.view(-1,self.state_out_size_final )
        # action_rep_2 = self.action_model_2(state_action_turn_pair)
        # action_rep_2 = self.action_model_2(state_rep)
        action_rep_2 = self.action_model_2(state_rep1)        
        action_rep = self.gumbel_connector.forward_ST_soft(action_rep_2, self.config.gumbel_temp)
        return state_rep, action_rep

class WoZGenerator_StateVae(BaseModel):
    def __init__(self, config):
        super(WoZGenerator_StateVae, self).__init__(config)
        
        state_in_size = config.state_noise_dim
        action_in_size = config.action_noise_dim
        turn_in_size = config.action_noise_dim
        self.state_out_size = config.vae_embed_size
        # self.action_out_size = 18
        self.action_out_size = 300
        self.gumble_length_index = self.gumble_index_multiwoz_binary()
        self.gumble_num = len(self.gumble_length_index)
        self.last_layers = nn.ModuleList()
        for gumbel_width in self.gumble_length_index:
            self.last_layers.append(nn.Linear(self.action_out_size, gumbel_width))
        

        self.state_model = nn.Sequential(
            # original: 5 block + 1 linear
            nn.Linear(state_in_size, 100),
            nn.ReLU(True),
            
            nn.Linear(100, self.state_out_size),
            nn.ReLU(True),

            nn.Linear(self.state_out_size, self.state_out_size),
        )
        # '''
        self.common_model = nn.Sequential(
            nn.Linear(state_in_size, state_in_size),
            nn.ReLU(True),
            
            nn.Linear(state_in_size, state_in_size),
            nn.ReLU(True),

        )


        self.action_model_2 = nn.Sequential(
            # nn.Linear( self.state_out_size, 96),            
            nn.Linear(state_in_size, 96),
            nn.ReLU(True),
            
            nn.Linear(96, 96),
            nn.ReLU(True),

            nn.Linear(96, self.action_out_size)
        )


    def gumble_index_multiwoz_binary(self):
        # index = 9 * [2]
        index = 1 * [300]
        return index

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()
        # self.clip_gradient()

    def forward(self, s_z):
        # state and action share one common MLP
        state_action_turn_pair = self.common_model(self.cast_gpu(s_z))
        # state_rep = torch.tanh(self.state_model(state_action_turn_pair))
        state_rep1 = self.state_model(state_action_turn_pair)
        # action_rep1 = self.action_model_2(state_rep1.detach())    
        action_rep1 = self.action_model_2(state_action_turn_pair)    
        input_to_gumble = []
        for layer, g_width in zip(self.last_layers, self.gumble_length_index):
            out = layer(action_rep1)
            out = self.gumbel_connector.forward_ST(out.view(-1,  g_width), self.config.gumbel_temp)
            input_to_gumble.append(out)
        action_rep = torch.cat(input_to_gumble, -1)
        return state_rep1, action_rep

class WoZDiscriminator(BaseModel):
    def __init__(self, config):
        super(WoZDiscriminator, self).__init__(config)
        self.state_in_size = config.vae_embed_size
        self.action_in_size = 300
        # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
        # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size/2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size/3)
        
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(self.state_in_size/2 + self.action_in_size/3, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(32, 1),

        )

    def decay_noise(self):
        self.noise_input *= 0.995
        
    def minibatch_averaging(self, inputs):
        """
        This method is explained in the MedGAN paper.
        """
        mean_per_feature = torch.mean(inputs, 0)
        mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
        return torch.cat((inputs, mean_per_feature_repeated), 1)

    def forward(self, state, action):
        s_z = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, state.shape))))
        s_a = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, action.shape))))
        # state_1 = self.state_rep(self.minibatch_averaging(self.cast_gpu(state)))
        # action_1 = self.action_rep(self.minibatch_averaging(self.cast_gpu(action)))
        # print(state.size(), action.size())
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        # print(state_1.size(), action_1.size())
        
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1-1e-7)
        return validity

    def forward_wgan(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state_1, action_1], 1)
        h = self.model(state_action)
        return h
    

######################################################################

class WoZDiscriminator_StateActionEmbed(BaseModel):
    def __init__(self, config):
        super(WoZDiscriminator_StateActionEmbed, self).__init__(config)
        self.state_in_size = config.vae_embed_size

        self.model = nn.Sequential(
            nn.Linear(self.state_in_size, self.state_in_size/2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(self.state_in_size/2, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(32, 1),
        )
    def decay_noise(self):
        pass
    
    def forward(self, state_action):
        state_action = self.cast_gpu(state_action)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1-1e-7)
        return validity



class WoZGenerator_StateActionEmbed(BaseModel):
    def __init__(self, config):
        super(WoZGenerator_StateActionEmbed, self).__init__(config)
        state_in_size = config.state_noise_dim
        self.state_out_size = config.vae_embed_size

        self.state_model = nn.Sequential(
            nn.Linear(state_in_size, 100),
            nn.ReLU(True),          

            nn.Linear(100, self.state_out_size),
            nn.ReLU(True),

            nn.Linear(self.state_out_size, self.state_out_size),
        )

        self.common_model = nn.Sequential(
            nn.Linear(state_in_size, state_in_size),
            nn.ReLU(True),
            
            nn.Linear(state_in_size, state_in_size),
            nn.ReLU(True),

        )

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()

    def forward(self, s_z):
        state_action_turn_pair = self.common_model(self.cast_gpu(s_z))
        state_rep = self.state_model(state_action_turn_pair)
        return state_rep


class WoZGenerator_StateVaeActionSeg(WoZGenerator_StateVae):
    # in this generator, the state rep is in the same continuous space with the vae output
    # the action rep is the concatenation of [domain, intent, slot], 1 * [10] + 1 * [14] + 28 * [2]
    def __init__(self, config):
        super(WoZGenerator_StateVaeActionSeg, self).__init__(config)
        state_in_size = config.state_noise_dim
        self.action_out_size = 48
        self.gumble_length_index = self.gumble_index_multiwoz_binary()
        self.gumble_num = len(self.gumble_length_index)
        self.last_layers = nn.ModuleList()
        for gumbel_width in self.gumble_length_index:
            self.last_layers.append(nn.Linear(self.action_out_size, gumbel_width))
        

        self.state_model = nn.Sequential(
            nn.Linear(state_in_size, 100),
            nn.ReLU(True),
            nn.Linear(100, self.state_out_size),
            nn.ReLU(True),
            nn.Linear(self.state_out_size, self.state_out_size),
        )
        self.common_model = nn.Sequential(
            nn.Linear(state_in_size, state_in_size),
            nn.ReLU(True),
            nn.Linear(state_in_size, state_in_size),
            nn.ReLU(True),
        )
        self.action_model_2 = nn.Sequential(
            # nn.Linear( self.state_out_size, 96),            
            nn.Linear(state_in_size, 96),
            # nn.BatchNorm1d(48),
            nn.ReLU(True),
            
            nn.Linear(96, 96),
            # nn.BatchNorm1d(48),
            nn.ReLU(True),

            nn.Linear(96, 48)
        )
    
    def gumble_index_multiwoz_binary(self):
        # This is for the action rep
        # index = 9 * [2]
        index = 1 * [10] + 1 * [14] + 28 * [2] + 1 * [10] + 1 * [14] + 28 * [2]
        return index


class WoZDiscriminator_StateVaeActionSeg(WoZDiscriminator):
    def __init__(self, config):
        super(WoZDiscriminator_StateVaeActionSeg, self).__init__(config)
        self.state_in_size = config.vae_embed_size
        self.action_in_size = 160
        # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
        # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size/2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size/3)
        
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            # nn.Linear(self.state_in_size/2, 64),
            nn.Linear(self.state_in_size/2 + self.action_in_size/3, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )
