# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch.autograd import Variable
import torch.nn as nn
import gan_model_sat
from gan_model import Discriminator, Generator, ContEncoder, VAE
from torch_utils import one_hot_embedding, LookupProb
from laed.dataset.corpora import PAD, EOS, EOT, BOS
from laed.utils import Pack, INT, FLOAT, LONG, cast_type
from utils import BCELoss_double, cal_accuracy
import torch.nn.functional as F

logger = logging.getLogger()

class GanRnnAgent(nn.Module):
    def __init__(self, corpus, config, action2name):
        super(GanRnnAgent, self).__init__()
        self.use_gpu = config.use_gpu
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.action2name=action2name
        self.lookupProb_ = LookupProb(action2name, config)
        # self.lookupProb_ = None       
        # self.generator = generator
        # self.discriminator = discriminator
        # self.cont_encoder = state_encoder
        self.context_encoder = ContEncoder(corpus, config)
        self.discriminator = Discriminator(config)
        self.generator = Generator(config)
        # FNN to get Y
        self.p_fc1 = nn.Linear(config.ctx_cell_size, config.ctx_cell_size)
        self.p_y = nn.Linear(config.ctx_cell_size, config.y_size * config.k)

        self.loss_BCE = nn.BCELoss()
        self.config = config

    def cast_gpu(self, var):
        if self.config.use_gpu:
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
        disc_v = self.discriminator(state_rep, action_rep)
        gen_loss = -torch.mean(torch.log(disc_v))
        gen_loss = Pack(gen_loss= gen_loss)
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



class GanAgent_SAT(GanRnnAgent):
    def __init__(self, corpus, config, action2name):
        super(GanAgent_SAT, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.Discriminator(config)
        self.generator = gan_model_sat.Generator(config)
        self.state_out_size = config.ctx_cell_size * config.bucket_num

    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], self.config.action_num)
        real_state_rep = self.np2var(batch_feed['state_table'], LONG)
        real_state_rep = one_hot_embedding(real_state_rep.view(-1), self.config.bucket_num).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_rep, action_rep = self.generator(z_noise)
        return state_rep, action_rep
    
class GanAgent_VAE(GanAgent_SAT):
    def __init__(self, corpus, config, action2name):
        super(GanAgent_VAE, self).__init__(corpus, config, action2name)
        self.discriminator =gan_model_sat.Discriminator(config)
        self.generator = gan_model_sat.Generator(config)
        self.state_out_size = config.ctx_cell_size * config.bucket_num
        self.vae = VAE(config)
        

    def read_real_data(self, sample_shape, batch_feed):
        action_data_feed = one_hot_embedding(batch_feed['action_id'], self.config.action_num)
        real_state_rep = self.np2var(batch_feed['state_table'], LONG)
        real_state_rep = one_hot_embedding(real_state_rep.view(-1), self.config.bucket_num).view(-1, self.state_out_size)
        return real_state_rep, action_data_feed

    def sample_step(self, sample_shape):
        batch_size, state_noise_dim, action_noise_dim = sample_shape
        z_noise = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, 1, (batch_size, state_noise_dim)))))
        state_rep, action_rep = self.generator(z_noise)
        return state_rep, action_rep
