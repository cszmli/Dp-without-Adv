# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch.autograd import Variable
import torch.nn as nn
import gan_model_sat, gan_model_vae
from gan_model import Discriminator, Generator, ContEncoder
from gan_model_vae import VAE, AutoEncoder, VAE_StateActionEmbed
from torch_utils import one_hot_embedding, LookupProb
from laed.dataset.corpora import PAD, EOS, EOT, BOS
from laed.utils import Pack, INT, FLOAT, LONG, cast_type
from utils import BCELoss_double, cal_accuracy
import torch.nn.functional as F
import random

logger = logging.getLogger()

class GanRnnAgent(nn.Module):
    def __init__(self, corpus, config, action2name):
        super(GanRnnAgent, self).__init__()
        self.use_gpu = config.use_gpu
        
        if config.state_type=='rnn':
            self.vocab = corpus.vocab
            self.rev_vocab = corpus.rev_vocab
            self.vocab_size = len(self.vocab)
            self.go_id = self.rev_vocab[BOS]
            self.eos_id = self.rev_vocab[EOS]
            self.context_encoder = ContEncoder(corpus, config)
            
        self.action2name=action2name
        self.lookupProb_ = LookupProb(action2name, config)
        self.discriminator = Discriminator(config)
        self.generator = Generator(config)

        self.loss_BCE = nn.BCELoss()
        self.config = config

    def cast_gpu(self, var):
        if self.config.use_gpu:
            return var.cuda()
        else:
            return var
        
    def binary2onehot(self,x):
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
    
    def shuffle_action(self, x):
        # len(x[0]) == 18   
        m_new = []
        for col in range(0, 18, 2):
            if np.random.random()>0.5:
                m_new.append(x[:,col])
                m_new.append(x[:,col+1])
            else:
                m_new.append(x[:,col+1])
                m_new.append(x[:,col])
        return torch.stack(m_new).transpose(1,0)

    
            

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        if type(inputs)==list:
            return cast_type(Variable(torch.Tensor(inputs)), dtype,
                         self.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.use_gpu)

    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        # a_z = Variable(torch.Tensor(np.random.normal(0, 1, (batch_size,  action_noise_dim))))
        # sample onhot tensor to represent the sampled actions
        # a_z = self.cast_gpu(Variable(torch.Tensor(np.eye(self.config.action_num)[np.random.choice(self.config.action_num, batch_size)])))
        state_rep, action_rep = self.generator(z_noise, z_noise)
        # print(torch.max(action_rep.data, dim=1))
        # return state_rep, z_noise
        return state_rep, action_rep


    def gen_train(self, sample_shape):
        state_rep, action_rep =self.sample_step(sample_shape)
        mean_dist_1 = - (state_rep.mean(dim=0) - state_rep).pow(2).mean() * 0.00003
        mean_dist_2 = - (action_rep.mean(dim=0) - action_rep).pow(2).mean() * 0.00003
        mean_dist = mean_dist_1 + mean_dist_2
        disc_v = self.discriminator(state_rep, action_rep)
        gen_loss = -torch.mean(torch.log(disc_v))
        gen_loss = Pack(gen_loss= gen_loss, mean_dist = mean_dist)
        return gen_loss, (state_rep.detach(), action_rep.detach())
    
    def gen_validate(self, sample_shape, record_gen=False):
        state_rep, action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return [], [state_rep.detach().cpu().tolist(), action_rep.detach().cpu().tolist()]
        else:
            return [], []
    
    
    def policy_validate_for_human(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], self.config.action_num)
        batch_size = sample_shape[0]
        real_state_rep = self.context_encoder(batch_feed).detach()
        policy_prob = self.policy_validate(real_state_rep, action_data_feed)
        return policy_prob.detach()
    
    def self_shuffle_disc_train(self, sample_shape, batch_feed):
        batch_size = sample_shape[0]
        real_state_rep, action_data_feed = self.read_real_data(sample_shape, batch_feed)
        fake_state_rep=real_state_rep[:,torch.randperm(real_state_rep.size()[1])]
        real_disc_v = self.discriminator(real_state_rep, action_data_feed)
        fake_disc_v = self.discriminator(fake_state_rep.detach(), action_data_feed.detach())

        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        return disc_loss, (fake_state_rep.detach(), action_data_feed.detach())


    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        real_state_rep, action_data_feed = self.read_real_data(sample_shape, batch_feed)
        real_disc_v = self.discriminator(real_state_rep, action_data_feed)
        if fake_state_action is None:
            fake_state_rep, fake_action_rep = self.sample_step(sample_shape)
        else:
            fake_state_rep, fake_action_rep = fake_state_action
        # if self.config.round_for_disc:
        #     fake_disc_v = self.discriminator(fake_state_rep.detach().round(), fake_action_rep.detach().round())
        # else:            
        if np.random.random()<0.5:
            fake_state_rep, fake_action_rep = real_state_rep.detach(), action_data_feed[:,torch.randperm(action_data_feed.size()[1])] 
        fake_disc_v = self.discriminator(fake_state_rep.detach(), fake_action_rep.detach())

        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc

    def policy_validate(self, state, action):
        sum_prob = []
        fc1_out =self.p_fc1(state)
        py_logits = self.p_y(torch.tanh(fc1_out)).view(-1, self.config.k)
        log_py = F.log_softmax(py_logits, dim=py_logits.dim()-1)
        #log_py = F.softmax(py_logits, dim=py_logits.dim()-1)
        log_py = log_py.view(-1, self.config.y_size, self.config.k)
        for log_py_line, action_line in zip(log_py, action):
            prob_line = self.lookupProb_(log_py_line, action_line)
            if type(prob_line)!=int and torch.isnan(prob_line):
                print(state[:2])
            sum_prob.append(prob_line)
        return torch.mean(torch.FloatTensor(sum_prob))

    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], self.config.action_num)
        if self.config.state_type=='rnn':
            real_state_rep = self.context_encoder(batch_feed).detach()
        elif self.config.state_type=='table':
            real_state_rep = self.np2var(batch_feed['state_table'], FLOAT)
        return real_state_rep, action_data_feed
    


class WGanAgent(GanRnnAgent):
    def __init__(self, corpus, config, action2name):
        super(WGanAgent, self).__init__(corpus, config, action2name)
    def gen_train(self, sample_shape):
        state_rep, action_rep =self.sample_step(sample_shape)
        disc_v = self.discriminator.forward_wgan(state_rep, action_rep)
        gen_loss = -torch.mean(disc_v)
        gen_loss = Pack(gen_loss= gen_loss)
        return gen_loss, (state_rep, action_rep)

    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        real_state_rep, action_data_feed = self.read_real_data(sample_shape, batch_feed)
        real_disc_v = self.discriminator.forward_wgan(real_state_rep, action_data_feed)
        if fake_state_action is None:
            fake_state_rep, fake_action_rep = self.sample_step(sample_shape)
        else:
            fake_state_rep, fake_action_rep = fake_state_action
        fake_disc_v = self.discriminator.forward_wgan(fake_state_rep.detach(), fake_action_rep.detach())

        real_disc_loss = - torch.mean(real_disc_v) 
        fake_disc_loss = torch.mean(fake_disc_v)
        disc_loss = real_disc_loss + fake_disc_loss
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = np.array([-real_disc_loss.item(), fake_disc_loss.item()])
        return disc_loss, disc_acc



    

class GanAgent_AutoEncoder(GanRnnAgent):
    def __init__(self, corpus, config, action2name):
        super(GanAgent_AutoEncoder, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_vae.Discriminator(config)
        self.generator = gan_model_vae.Generator(config)
        self.state_out_size = config.state_out_size
        self.vae = gan_model_vae.AutoEncoder(config)
        # self.vae_in_size = config.state_out_size + config.action_num
        self.autoencoder_in_size = config.state_out_size + 9
        self.config = config 
        
        
    def vae_train(self, batch_feed):
        state_rep, action_rep = self.read_real_data(None, batch_feed)
        state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        recon_batch = self.vae(state_action)
        loss = self.autoencoder_loss(recon_batch, state_action)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss
        
        
    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = self.np2var(batch_feed['action_id_binary'], FLOAT).view(-1, 9)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed
    
    def autoencoder_loss(self, recon_x, x):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).mean()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        l2_loss = self.vae.l2_norm()
        loss = BCE + l2_loss
        return loss + l2_loss
    
    
    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_action_rep = self.generator(z_noise, z_noise)
        discrete_state_action_rep = self.get_decode_result(state_action_rep)
        # state_rep, action_rep = torch.split(discrete_state_action_rep, 392, dim=1)       
        state_rep, action_rep = discrete_state_action_rep[:,:self.state_out_size].clone(), discrete_state_action_rep[:,self.state_out_size:].clone()      
        return state_rep, action_rep

    def get_decode_result(self, state_action_rep):
        result = self.vae.decode(state_action_rep)
        return result
    
    def gan_vae_optimizer(self, config):
        params = list(self.generator.parameters()) + list(self.vae.parameters())
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           params), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(params, lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(params, lr=config.init_lr,
                                       momentum=config.momentum)


class GanAgent_AutoEncoder_Encode(GanAgent_AutoEncoder):
    def __init__(self, corpus, config, action2name):
        super(GanAgent_AutoEncoder_Encode, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_vae.Discriminator(config)
        self.autoencoder_in_size = config.state_out_size + 300
    def vae_train(self, batch_feed):
        state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        recon_batch = self.vae(state_action)
        loss = self.autoencoder_loss(recon_batch, state_action)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss
        
        
    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = self.np2var(batch_feed['action_id_binary'], FLOAT).view(-1, 9)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def read_real_data_onehot(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], 300)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed
    
    def autoencoder_loss(self, recon_x, x):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).mean()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        l2_loss = self.vae.l2_norm()
        loss = BCE + l2_loss
        return loss + l2_loss
    
    
    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_action_rep = self.generator(z_noise, z_noise)
        return state_action_rep

    def get_vae_embed(self, state_action_rep):
        result = self.vae.encode(state_action_rep)
        return result
    
    def gan_vae_optimizer(self, config):
        params = list(self.generator.parameters()) + list(self.vae.parameters())
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           params), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(params, lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(params, lr=config.init_lr,
                                       momentum=config.momentum)
            
    def gen_train(self, sample_shape):
        state_action_rep =self.sample_step(sample_shape)
        mean_dist = - (state_action_rep.mean(dim=0) - state_action_rep).pow(2).sum() * 0.00003
        disc_v = self.discriminator(state_action_rep)
        gen_loss = -torch.mean(torch.log(disc_v))
        gen_loss = Pack(gen_loss= gen_loss, mean_dist = mean_dist)
        return gen_loss, state_action_rep.detach()
        
    
    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        # state_rep, action_rep = self.read_real_data(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        embed_batch = self.vae.encode(state_action)
        real_disc_v = self.discriminator(embed_batch.detach())
        if fake_state_action is None:
            fake_state_action = self.sample_step(sample_shape)
        else:
            fake_state_action = fake_state_action
        if self.config.round_for_disc:
            fake_disc_v = self.discriminator(fake_state_action.detach())
        else:            
            fake_disc_v = self.discriminator(fake_state_action.detach())

        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc

    def gen_validate(self, sample_shape, record_gen=False):
        state_action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return [], state_action_rep
        else:
            return [], []
    


class GanAgent_AutoEncoder_State(GanAgent_AutoEncoder):
    # only state is fed to autoencoder
    def __init__(self, corpus, config, action2name):
        super(GanAgent_AutoEncoder_State, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.WoZDiscriminator(config)
        self.generator =gan_model_sat.WoZGenerator_StateVae(config)
        self.autoencoder_in_size = config.state_out_size 
        self.vae = gan_model_vae.AutoEncoder(config)

    def vae_train(self, batch_feed):
        # state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_rep = self.cast_gpu(state_rep)
        recon_batch = self.vae(state_rep)
        loss = self.autoencoder_loss(recon_batch, state_rep)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss
        
        
    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = self.np2var(batch_feed['action_id_binary'], FLOAT).view(-1, 9)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def read_real_data_onehot(self, sample_shape, batch_feed):
        action_id = self.binary2onehot(batch_feed['action_id_binary'])
        action_data_feed = self.np2var(action_id, FLOAT).view(-1, 18)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def read_real_data_onehot_300(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], 300)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed
    
    def autoencoder_loss(self, recon_x, x):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).mean()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        l2_loss = self.vae.l2_norm()
        loss = BCE + l2_loss
        return loss + l2_loss
    
    
    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_rep, action_rep = self.generator(z_noise)
        return state_rep, action_rep

    def get_vae_embed(self, state_action_rep):
        result = self.vae.encode(state_action_rep)
        return result
    
    def gan_vae_optimizer(self, config):
        params = list(self.generator.parameters()) + list(self.vae.parameters())
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            params), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(params, lr=config.init_lr,
                                    momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(params, lr=config.init_lr,
                                        momentum=config.momentum)
            
    def gen_train(self, sample_shape):
        state_rep, action_rep =self.sample_step(sample_shape)
        mean_dist_1 = - (state_rep.mean(dim=0) - state_rep).pow(2).mean() * self.config.sim_factor
        mean_dist_2 = - (action_rep.mean(dim=0) - action_rep).pow(2).mean() * self.config.sim_factor
        mean_dist = mean_dist_1 + mean_dist_2
        # mean_dist =  mean_dist_2
        disc_v = self.discriminator(state_rep, action_rep)
        gen_loss = -torch.mean(torch.log(disc_v))
        gen_loss = Pack(gen_loss= gen_loss, mean_dist = mean_dist)
        return gen_loss, (state_rep.detach(), action_rep.detach())
        
    
    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        # state_rep, action_rep = self.read_real_data(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        # state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        embed_batch = self.vae.get_embed(self.cast_gpu(state_rep))
        real_disc_v = self.discriminator(embed_batch.detach(), self.cast_gpu(action_rep))
        if fake_state_action is None:
            fake_state_action = self.sample_step(sample_shape)
        else:
            fake_state_action = fake_state_action
        state_rep_f, action_rep_f = fake_state_action
        if np.random.random()<0.5:
            # print(action_rep.size())
            # state_rep_f, action_rep_f = embed_batch.detach(), self.shuffle_action(action_rep)
            state_rep_f, action_rep_f = embed_batch.detach(), action_rep[:,torch.randperm(action_rep.size()[1])] 
            # print(action_rep_f.size())
        fake_disc_v = self.discriminator(state_rep_f.detach(), action_rep_f.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc

    def gen_validate(self, sample_shape, record_gen=False):
        state_rep, action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return state_rep, action_rep
        else:
            return [], []
    
class GanAgent_VAE_State(GanAgent_AutoEncoder_State):
    # only state is fed to VAE, action is onehot with 300 dims
    def __init__(self, corpus, config, action2name):
        super(GanAgent_VAE_State, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.WoZDiscriminator(config)
        self.generator =gan_model_sat.WoZGenerator_StateVae(config)
        self.autoencoder_in_size = config.state_out_size 
        self.vae = gan_model_vae.VAE(config)

    def vae_train(self, batch_feed):
        # state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_rep = self.cast_gpu(state_rep)
        recon_batch, mu, logvar = self.vae(state_rep)
        loss = self.vae_loss(recon_batch, state_rep, mu, logvar)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss

    
    def vae_loss(self, recon_x, x, mu, logvar):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size), reduction='sum')
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l2_loss = self.vae.l2_norm()
        loss = (BCE + KLD)/len(x.view(-1))
        return loss + l2_loss

    def get_vae_embed(self, state_action_rep):
        mean, _ = self.vae.encode(state_action_rep)
        return mean
    

   
class WGanAgent_VAE_State(GanAgent_AutoEncoder_State):
    # only state is fed to VAE, action is onehot with 300 dims
    def __init__(self, corpus, config, action2name):
        super(WGanAgent_VAE_State, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.WoZDiscriminator(config)
        self.generator =gan_model_sat.WoZGenerator_StateVae(config)
        self.autoencoder_in_size = config.state_out_size 
        self.vae = gan_model_vae.VAE(config)
        
    def gen_train(self, sample_shape):
        state_rep, action_rep =self.sample_step(sample_shape)
        disc_v = self.discriminator.forward_wgan(state_rep, action_rep)
        gen_loss = -torch.mean(disc_v)
        gen_loss = Pack(gen_loss= gen_loss)
        return gen_loss, (state_rep.detach(), action_rep.detach())

    def gen_validate(self, sample_shape, record_gen=False):
        state_rep, action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return state_rep, action_rep
        else:
            return [], []

    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        embed_batch = self.vae.get_embed(self.cast_gpu(state_rep))
        real_disc_v = self.discriminator.forward_wgan(embed_batch.detach(), self.cast_gpu(action_rep))

        if fake_state_action is None:
            fake_state_rep, fake_action_rep = self.sample_step(sample_shape)
        else:
            fake_state_rep, fake_action_rep = fake_state_action
        fake_disc_v = self.discriminator.forward_wgan(fake_state_rep.detach(), fake_action_rep.detach())

        real_disc_loss = - torch.mean(real_disc_v) 
        fake_disc_loss = torch.mean(fake_disc_v)
        disc_loss = real_disc_loss + fake_disc_loss
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = np.array([-real_disc_loss.item(), fake_disc_loss.item()])
        return disc_loss, disc_acc


    def vae_train(self, batch_feed):
        # state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_rep = self.cast_gpu(state_rep)
        recon_batch, mu, logvar = self.vae(state_rep)
        loss = self.vae_loss(recon_batch, state_rep, mu, logvar)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss

    
    def vae_loss(self, recon_x, x, mu, logvar):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size), reduction='sum')
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l2_loss = self.vae.l2_norm()
        loss = (BCE + KLD)/len(x.view(-1))
        return loss + l2_loss

    def get_vae_embed(self, state_action_rep):
        mean, _ = self.vae.encode(state_action_rep)
        return mean

##################################################################
########### In this agent, the state and action are fed to the VAE together.
class GanAgent_VAE_StateActioneEmbed(GanAgent_AutoEncoder):
    def __init__(self, corpus, config, action2name):
        super(GanAgent_VAE_StateActioneEmbed, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.WoZDiscriminator_StateActionEmbed(config)
        self.generator =gan_model_sat.WoZGenerator_StateActionEmbed(config)
        self.autoencoder_in_size = config.state_out_size + 100
        # self.vae = gan_model_vae.VAE_StateActionEmbed(config) # the input size is 392 + 100
        self.vae = gan_model_vae.VAE_StateActionEmbedMerged(config) # the input size is 392 + 100



    def read_real_data_onehot_300(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], 300)
        action_rep_seg = self.np2var(batch_feed['action_rep_seg'], FLOAT).view(-1, 100)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_rep_seg

    def vae_train(self, batch_feed):
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_action_rep = torch.cat([state_rep, action_rep],-1)
        recon_batch, mu, logvar = self.vae(state_action_rep)
        loss = self.vae_loss(recon_batch, state_action_rep, mu, logvar)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss

    
    def vae_loss(self, recon_x, x, mu, logvar):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size), reduction='sum')
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l2_loss = self.vae.l2_norm()
        loss = (BCE + KLD)/len(x.view(-1))
        return loss + l2_loss

    def get_vae_embed(self, state_action_rep):
        mean, _ = self.vae.encode(state_action_rep)
        return mean

    
    def autoencoder_loss(self, recon_x, x):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size))
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).mean()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        l2_loss = self.vae.l2_norm()
        loss = BCE + l2_loss
        return loss + l2_loss
    
    
    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_action_rep= self.generator(z_noise)
        return state_action_rep

    
    def gan_vae_optimizer(self, config):
        params = list(self.generator.parameters()) + list(self.vae.parameters())
        if config.op == 'adam':
            print("Use adam")
            return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                           params), lr=config.init_lr, betas=(0.5, 0.999))
        elif config.op == 'sgd':
            print("Use SGD")
            return torch.optim.SGD(params, lr=config.init_lr,
                                   momentum=config.momentum)
        elif config.op == 'rmsprop':
            print("RMSProp")
            return torch.optim.RMSprop(params, lr=config.init_lr,
                                       momentum=config.momentum)
        
    def gen_train(self, sample_shape):
        state_action_rep =self.sample_step(sample_shape)
        mean_dist_1 = - (state_action_rep.mean(dim=0) - state_action_rep).pow(2).mean() * self.config.sim_factor
        mean_dist = mean_dist_1 
        disc_v = self.discriminator(state_action_rep)
        gen_loss = -torch.mean(torch.log(disc_v))
        gen_loss = Pack(gen_loss= gen_loss, mean_dist = mean_dist)
        return gen_loss, state_action_rep.detach()
    
        
    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        embed_batch = self.vae.get_embed(self.cast_gpu(state_action))
        real_disc_v = self.discriminator(embed_batch.detach())
        if fake_state_action is None:
            fake_state_action = self.sample_step(sample_shape)
        else:
            fake_state_action = fake_state_action
        state_rep_f = fake_state_action
        fake_disc_v = self.discriminator(state_rep_f.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc
    
    def gen_validate(self, sample_shape, record_gen=False):
        state_action_rep =self.sample_step(sample_shape)
        # policy_prob = self.policy_validate(state_rep, action_rep)
        if record_gen:
            return [], state_action_rep
        else:
            return [], []

###########################################################
###########   
###########################################################
class GanAgent_StateVaeActionSeg(GanAgent_AutoEncoder_State):
    # only state is fed to VAE, the action is the concatenation with (domain, act, slot), rather than onehot-300
    def __init__(self, corpus, config, action2name):
        super(GanAgent_StateVaeActionSeg, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.WoZDiscriminator_StateVaeActionSeg(config)
        self.generator =gan_model_sat.WoZGenerator_StateVaeActionSeg(config)
        self.autoencoder_in_size = config.state_out_size 
        self.vae = gan_model_vae.VAE(config)

    def read_real_data_onehot_300(self, sample_shape, batch_feed):
        # the action rep should be the concatenated version, which has dimension 160
        action_data_feed = self.np2var(batch_feed['action_rep_seg'], FLOAT).view(-1, 160)
        real_state_rep = self.np2var(batch_feed['state_convlab'], FLOAT).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def vae_train(self, batch_feed):
        # state_rep, action_rep = self.read_real_data_onehot(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        state_rep = self.cast_gpu(state_rep)
        recon_batch, mu, logvar = self.vae(state_rep)
        loss = self.vae_loss(recon_batch, state_rep, mu, logvar)
        vae_loss = Pack(vae_loss=loss) 
        return vae_loss
    

    
    def vae_loss(self, recon_x, x, mu, logvar):
        if self.config.vae_loss=='bce':
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.autoencoder_in_size), reduction='sum')
        elif self.config.vae_loss=='mean_sq':
            BCE = 0.5 * (x - recon_x).pow(2).sum()
        else:
            raise ValueError("No such vae loss type: {}".format(self.config.vae_loss))
        
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        l2_loss = self.vae.l2_norm()
        loss = (BCE + KLD)/len(x.view(-1))
        return loss + l2_loss

    def get_vae_embed(self, state_action_rep):
        mean, _ = self.vae.encode(state_action_rep)
        return mean

    def disc_train(self, sample_shape, batch_feed, fake_state_action=None):
        batch_size = sample_shape[0]
        # state_rep, action_rep = self.read_real_data(None, batch_feed)
        state_rep, action_rep = self.read_real_data_onehot_300(None, batch_feed)
        # state_action = torch.cat([self.cast_gpu(state_rep), self.cast_gpu(action_rep)], -1)
        embed_batch = self.vae.get_embed(self.cast_gpu(state_rep))
        real_disc_v = self.discriminator(embed_batch.detach(), self.cast_gpu(action_rep))
        if fake_state_action is None:
            fake_state_action = self.sample_step(sample_shape)
        else:
            fake_state_action = fake_state_action
        state_rep_f, action_rep_f = fake_state_action
        if np.random.random()<0.5:
            # print(action_rep.size())
            # state_rep_f, action_rep_f = embed_batch.detach(), self.shuffle_action(action_rep)
            state_rep_f, action_rep_f = embed_batch.detach(), action_rep[torch.randperm(action_rep.size()[0]), :] 
            # print(action_rep_f.size())
        fake_disc_v = self.discriminator(state_rep_f.detach(), action_rep_f.detach())
        rf_disc_v = torch.cat([real_disc_v, fake_disc_v], dim=0)
        labels_one = torch.cat([torch.full((batch_size,),1.0), torch.full((batch_size,),0.0)])
        labels_one = self.cast_gpu(labels_one)
        # labels_one = self.cast_gpu(torch.FloatTensor([1] * batch_size + [0] * batch_size))
        disc_loss = self.loss_BCE(rf_disc_v.view(-1), labels_one)
        # disc_loss = BCELoss_double(rf_disc_v.view(-1), labels)
        # disc_loss = - torch.mean(real_disc_v) + torch.mean(fake_disc_v)
        disc_loss = Pack(disc_loss=disc_loss)
        disc_acc = cal_accuracy(rf_disc_v.view(-1), labels_one)
        return disc_loss, disc_acc
    
