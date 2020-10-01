# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch.autograd import Variable
import torch.nn as nn
import json
import torch.nn.functional as F
import copy
import sys

INT = 0
LONG = 1
FLOAT = 2
def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var

def load_reward_model(agent, pre_sess_path, use_gpu):
    print(pre_sess_path)
    if not os.path.isfile(pre_sess_path) and not os.path.isdir(pre_sess_path):
    # if not os.path.isdir(pre_sess_path):    
        raise ValueError("No reward model was loaded")
    else:
        reward_path = os.path.join(pre_sess_path, "model_lirl")
        if use_gpu:
            reward_sess = torch.load(reward_path)
        else:
            reward_sess = torch.load(reward_path, map_location='cpu')
        agent.discriminator.load_state_dict(reward_sess['discriminator'])
        
        # vae_path = os.path.join(pre_sess_path, "model_vae")
        # vae_sess = torch.load(vae_path)
        # agent.vae.load_state_dict(vae_sess['vae'])
        agent.vae.load_state_dict(reward_sess['vae'])
        
        print("Loading reward model finished!")

def binary2onehot(x):
    batch_size, digit_num = len(x), len(x[0])
    if digit_num != 9:
        raise ValueError("check the binary length and the current one is {}".format(digit_num))
    one_hot_matrix = []
    for line in x:
        one_hot = []
        for v in line:
            if v==0:
                one_hot+=[1,0]
            elif v==1:
                one_hot+=[0,1]
            else:
                raise ValueError("illegal onehot input: {}".format(v))
        one_hot_matrix.append(one_hot)
    return one_hot_matrix


def one_hot_embedding(labels, num_classes):
    # print(labels)
    if type(labels)==list:
        labels = torch.LongTensor(labels)
    y = torch.eye(num_classes) 
    return y[labels]  
    

class RewardAgent(nn.Module):
    def __init__(self, use_gpu):
        super(RewardAgent, self).__init__()
        config = None
        self.use_gpu = use_gpu
        self.discriminator = Discriminator(self.use_gpu)
        self.vae = AutoEncoder(self.use_gpu)

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
    
    def _int2binary_9(self,x):
        return list(reversed( [(x >> i) & 1 for i in range(9)]))

    def forward(self,batch_feed):        
        state = batch_feed['states']
        action = batch_feed['actions']
        action_list = action.view(-1).tolist()
        action_binary = []
        for act in action_list:
            action_binary.append(self._int2binary_9(int(act)))
        reward = self.discriminator(state, self.np2var(action_binary,FLOAT))
        return reward
        
class RewardAgent_EncoderSide(nn.Module):
    def __init__(self, use_gpu=False, vae_type='autoencoder', update=False, real_data_feed=None):
        super(RewardAgent_EncoderSide, self).__init__()
        config = None
        self.use_gpu = use_gpu
        # self.discriminator = Discriminator_SA(self.use_gpu)
        if not update:
            self.discriminator = WoZDiscriminator(self.use_gpu)
        else:
            self.discriminator = WoZDiscriminator_Update(real_data_feed,self.use_gpu, 16)
        
        if vae_type=='autoencoder':
            self.vae = AutoEncoder(self.use_gpu)
        elif vae_type=='vae':
            self.vae = VAE(self.use_gpu)
        else:
            raise ValueError("no such vae type {}".format(vae_type))

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
        
    def _int2binary_9(self,x):
        return list(reversed( [(x >> i) & 1 for i in range(9)]))

    def get_action_rep(self, action_list):
        return one_hot_embedding(action_list, 300)

    def forward(self,batch_feed):        
        state = batch_feed['states']
        action = batch_feed['actions']
        # print(action)
        action_list = action.view(-1).tolist()
        # action_binary = []
        # for act in action_list:
        #     action_binary.append(self._int2binary_9(int(act)))
        # action_binary_onehot = binary2onehot(action_binary)
        # action_data_feed = self.np2var(action_binary_onehot, FLOAT).view(-1, 18)
        action_data_feed = self.get_action_rep(action_list)
        # state_action = torch.cat([self.cast_gpu(state), self.np2var(action_binary,FLOAT)], -1)
        state_action = self.cast_gpu(state)
        embed_rep = self.vae.get_embed(state_action)
        reward = self.discriminator(embed_rep, action_data_feed)
        return reward.detach()
    
    def forward_validate(self,batch_feed):        
        state = batch_feed['state_convlab']
        action = batch_feed['action_id']
        action_list = action
        # action_binary = []
        # for act in action_list:
        #     action_binary.append(self._int2binary_9(int(act)))
        # action_binary_onehot = binary2onehot(action_binary)
        # action_data_feed = self.np2var(action_binary_onehot, FLOAT).view(-1, 18)
        action_data_feed = self.get_action_rep(action_list)

        state_action = self.np2var(state,FLOAT)
        embed_rep = self.vae.get_embed(state_action)
        reward = self.discriminator(embed_rep, action_data_feed)
        return reward
    
    def update(self, fake_batch_feed):
        return self.discriminator.disc_train(self.vae, fake_batch_feed)
    


class RewardAgent_StateVaeActionSeg(RewardAgent_EncoderSide):
    def __init__(self, use_gpu=False, vae_type='autoencoder'):
        super(RewardAgent_StateVaeActionSeg, self).__init__(use_gpu, vae_type)
        self.discriminator = WoZDiscriminator_StateVaeActionSeg(use_gpu)
        action_rep_path = './data/multiwoz/action_rep_seg.json'
        if not os.path.isfile(action_rep_path):
            raise ValueError("No action rep was loaded")
        with open(action_rep_path, 'r') as f:
            action_rep_seg = json.load(f)
            assert len(action_rep_seg)==300 
        self.action_rep_seg = self.np2var(action_rep_seg, FLOAT)
    
    def get_action_rep(self, action_list):
        if type(action_list)==list:
            act_index = torch.LongTensor(action_list)
        else:
            act_index = action_list
        return self.action_rep_seg[act_index]
        

        
#########################################################################
###########   The following parts are for the Neural Networks  ##########
#########################################################################
class BaseModel(nn.Module):
    def __init__(self, use_gpu):
        super(BaseModel, self).__init__()
        self.use_gpu = use_gpu
        self.flush_valid = False

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


class Discriminator(BaseModel):
    def __init__(self, use_gpu):
        super(Discriminator, self).__init__(use_gpu)
        dropout = 0.3
        self.state_in_size = 392
        self.action_in_size = 9
        self.state_rep = nn.Linear(self.state_in_size, int(self.state_in_size/2))
        self.action_rep = nn.Linear(self.action_in_size, int(self.action_in_size/2))
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(int(self.state_in_size/2 + self.action_in_size/2), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        # print(state.shape, action_1.shape)
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-6, 1-1e-6)
        return validity
    
    
      
class AutoEncoder(BaseModel):
    def __init__(self, use_gpu):
        super(AutoEncoder, self).__init__(use_gpu)
        self.use_gpu = use_gpu
        # self.vae_in_size = 392 + 300
        self.vae_in_size = 392
        self.vae_embed_size =64
        dropout = 0.3
        
        self.encode_model = nn.Sequential(
            nn.Dropout(dropout),          
            nn.Linear(self.vae_in_size, int(self.vae_in_size/2)),
            nn.Tanh(),
            nn.Dropout(dropout),                      
            nn.Linear(int(self.vae_in_size/2), self.vae_embed_size),
            nn.Tanh(), 
        )
        self.decode_model = nn.Sequential(
            nn.Dropout(dropout),                  
            nn.Linear(self.vae_embed_size, int(self.vae_in_size/2)),
            nn.Sigmoid(),
            nn.Dropout(dropout),                      
            nn.Linear(int(self.vae_in_size/2), self.vae_in_size),
            nn.Sigmoid(),
        )
        
    def get_embed(self, x):
        return self.encode(x)

    def encode(self, x):
        h = self.encode_model(x)
        return h

    def decode(self, z):
        h = self.decode_model(z)
        return h

    def forward(self, x):
        x = self.cast_gpu(x)
        z = self.encode(x.view(-1, self.vae_in_size))
        return self.decode(z)

      
class VAE(BaseModel):
    def __init__(self, use_gpu):
        super(VAE, self).__init__(use_gpu)
        self.use_gpu = use_gpu
        self.vae_in_size = 392
        self.vae_embed_size =64

        self.encode_model = nn.Sequential(
            nn.Linear(self.vae_in_size, self.vae_in_size//4),
            nn.ReLU(True),    
        )
        self.decode_model = nn.Sequential(
            nn.Linear(self.vae_embed_size, self.vae_in_size//4),
            nn.ReLU(True),
            nn.Linear(self.vae_in_size//4, self.vae_in_size),
        )
        
        
        self.fc21 = nn.Linear(self.vae_in_size//4, self.vae_embed_size)
        self.fc22 = nn.Linear(self.vae_in_size//4, self.vae_embed_size)

            
    def get_embed(self, x):
        mean, _ = self.encode(x)
        return mean

    def encode(self, x):
        h = self.encode_model(x)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.decode_model(z)
        return torch.sigmoid(h)
        # return h 

    def forward(self, x):
        x = self.cast_gpu(x)
        mu, logvar = self.encode(x.view(-1, self.vae_in_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    


class Discriminator_SA(BaseModel):
    def __init__(self, use_gpu):
        super(Discriminator_SA, self).__init__(use_gpu)
        dropout = 0.3
        self.input_size = 64
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 1)
        )

    def decay_noise(self):
        self.noise_input *= 0.995

    def forward(self, state_action):
        validity = torch.sigmoid(self.model(self.cast_gpu(state_action)))
        validity = torch.clamp(validity, 1e-6, 1-1e-6)
        return validity


class WoZDiscriminator(BaseModel):
    def __init__(self,use_gpu):
        super(WoZDiscriminator, self).__init__(use_gpu)
        dropout = 0.3
        self.state_in_size = 64
        self.action_in_size = 300
        # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
        # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)
        
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(self.state_in_size//2 + self.action_in_size//3, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )

    def forward(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1-1e-7)
        return validity
    



class WoZDiscriminator_Update(BaseModel):
    def __init__(self,real_data_feed, use_gpu, batch_size):
        super(WoZDiscriminator_Update, self).__init__(use_gpu)
        dropout = 0.3
        self.batch_size = batch_size
        self.real_data_feed = real_data_feed
        self.loss_BCE = nn.BCELoss()
        self.state_in_size = 64
        self.action_in_size = 300
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)
        
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(self.state_in_size//2 + self.action_in_size//3, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
        )
    def get_optimizer(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=0.0005, betas=(0.5, 0.999))


    def forward(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1-1e-7)
        return validity
    
    def sample_real_batch(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        action = one_hot_embedding(action, 300)
        return self.np2var(state, FLOAT), action
    

    def disc_train(self, vae, fake_batch_feed):
        state_rep, action_rep = self.sample_real_batch()
        embed_batch = vae.get_embed(self.cast_gpu(state_rep))
        real_disc_v = self.forward(embed_batch.detach(), self.cast_gpu(action_rep))

        fake_state = cast_type(fake_batch_feed['states'], FLOAT, False)
        fake_state = vae.get_embed(self.cast_gpu(fake_state))
        fake_action =  cast_type(fake_batch_feed['actions'], LONG, False)
        fake_size = len(fake_state)
        fake_action = one_hot_embedding(fake_action, 300)
        
        fake_disc_v = self.forward(fake_state.detach(), fake_action.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((self.batch_size,),1.0), torch.full((fake_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        return disc_loss


class WoZDiscriminator_StateVaeActionSeg(WoZDiscriminator):
    def __init__(self, use_gpu):
        super(WoZDiscriminator_StateVaeActionSeg, self).__init__(use_gpu)
        self.state_in_size = 64
        self.action_in_size = 160
        dropout = 0.3
        # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
        # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)
        
        self.noise_input = 0.01
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            nn.Linear(self.state_in_size//2 + self.action_in_size//3, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )


class A2C_Discriminator(BaseModel):
    def __init__(self, use_gpu, real_data_feed, batch_size):
        super(A2C_Discriminator, self).__init__(use_gpu)
        self.real_data_feed = real_data_feed
        self.batch_size = batch_size
        dropout = 0.3
        self.loss_BCE = nn.BCELoss()
        self.state_in_size = 392
        self.action_in_size = 300
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)

        self.model = nn.Sequential(
            nn.Linear(self.state_in_size + self.action_in_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    
    def get_optimizer(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=0.0005, betas=(0.5, 0.999))

    def sample_real_batch(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        action = one_hot_embedding(action, 300)
        return self.np2var(state, FLOAT), action

    def sample_real_batch_id(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        return self.np2var(state, FLOAT), self.np2var(action, INT)


    def forward(self, state, action):
        # state_1 = self.state_rep(self.cast_gpu(state))
        # action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state, action], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1-1e-7)
        return validity

    def get_reward(self, batch_feed):
        state = cast_type(batch_feed['states'], FLOAT, False)
        action = cast_type(batch_feed['actions'], LONG, False)
        action = one_hot_embedding(action, 300)
        fake_disc_v = self.forward(state.detach(), action.detach())
        return fake_disc_v.detach().view(-1)
      
    def disc_train(self, fake_batch_feed):
        # batch
        real_state, real_action= self.sample_real_batch()
        real_disc_v = self.forward(real_state, real_action)

        fake_state = cast_type(fake_batch_feed['states'], FLOAT, False)
        fake_action =  cast_type(fake_batch_feed['actions'], LONG, False)
        fake_size = len(fake_state)
        fake_action = one_hot_embedding(fake_action, 300)
        # print(len(real_state), len(fake_state))
        # assert len(real_state)==len(fake_state)

        fake_disc_v = self.forward(fake_state.detach(), fake_action.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((self.batch_size,),1.0), torch.full((fake_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        return disc_loss



class AIRL(BaseModel):
    def __init__(self, use_gpu, real_data_feed, batch_size):
        super(AIRL, self).__init__(use_gpu)
        self.real_data_feed = real_data_feed
        self.batch_size = batch_size
        dropout = 0.3
        self.loss_BCE = nn.BCELoss()
        self.state_in_size = 392
        self.action_in_size = 300
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size//2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size//3)

        self.model_g = nn.Sequential(
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.state_in_size + self.action_in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.model_h = nn.Sequential(
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.state_in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    
    def get_optimizer(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           self.parameters()), lr=0.0005, betas=(0.5, 0.999))

    def sample_real_batch(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        action = one_hot_embedding(action, 300)
        next_state = batch['state_convlab_next']
        return self.np2var(state, FLOAT), action, self.np2var(next_state, FLOAT)
    
    def sample_real_batch_id(self):
        batch = self.real_data_feed.next_batch()
        if batch is None:
            self.real_data_feed.epoch_init(shuffle=True, verbose=True)
            batch = self.real_data_feed.next_batch()
        state = batch['state_convlab']
        action = batch['action_id']
        return self.np2var(state, FLOAT), self.np2var(action, INT)


    def forward(self, state, action, state_next):
        # state_1 = self.state_rep(self.cast_gpu(state))
        # action_1 = self.action_rep(self.cast_gpu(action))
        # state_2 = self.state_rep(self.cast_gpu(state_next))

        state_1 = self.cast_gpu(state)
        action_1 = self.cast_gpu(action)
        state_2 = self.cast_gpu(state_next)

        state_action = torch.cat([state_1, action_1], -1)
        validity = self.model_g(state_action) + 0.99 * self.model_h(state_2) - self.model_h(state_1)
        return validity

    def get_reward(self, batch_feed):
        state = cast_type(batch_feed['states'], FLOAT, False)
        action = cast_type(batch_feed['actions'], LONG, False)
        action = one_hot_embedding(action, 300)
        state_next = cast_type(batch_feed['next_states'], FLOAT, False)
        fake_disc_v = self.forward(state.detach(), action.detach(), state_next.detach())
        return fake_disc_v.detach().view(-1)
      
    def disc_train(self, fake_batch_feed):
        # batch
        real_state, real_action, real_state_next= self.sample_real_batch()
        real_disc_v = self.forward(real_state, real_action, real_state_next)

        fake_state = cast_type(fake_batch_feed['states'], FLOAT, False)
        fake_action =  cast_type(fake_batch_feed['actions'], LONG, False)
        fake_state_next = cast_type(fake_batch_feed['next_states'], FLOAT, False)
        fake_size = len(fake_state)
        fake_action = one_hot_embedding(fake_action, 300)

        fake_disc_v = self.forward(fake_state.detach(), fake_action.detach(), fake_state_next.detach())
        loss = - real_disc_v.mean() + fake_disc_v.mean()
        return loss