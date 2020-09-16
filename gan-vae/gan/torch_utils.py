# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import json
import logging
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch
from nltk.tokenize.moses import MosesDetokenizer
# from mosestokenizer import MosesDetokenizer
import nltk
import sys
from collections import defaultdict, deque
from argparse import Namespace
import numpy as np
from laed.utils import cast_type
from torch.autograd import Variable
import random
import copy

INT = 0
LONG = 1
FLOAT = 2
logger = logging.getLogger()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def adjust_learning_rate(optimizer, epoch, config):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = config.init_lr * (0.7 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def one_hot_embedding(labels, num_classes):
    # print(labels)
    if type(labels)==list:
        labels = torch.LongTensor(labels)
    y = torch.eye(num_classes) 
    return y[labels] 


class GumbelConnector(nn.Module):
    def __init__(self, use_gpu):
        super(GumbelConnector, self).__init__()
        self.use_gpu = use_gpu

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = Variable(-torch.log(-torch.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def soft_argmax(self, logits, temperature=0.2):
        return F.softmax(logits / temperature, dim=logits.dim()-1)

    def forward(self, logits, temperature=1.0, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :param return_max_id
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        _, y_hard = torch.max(y, dim=1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(torch.zeros(y.size())), FLOAT, self.use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y
    
    def forward_ST(self, logits, temperature=0.8):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def forward_ST_soft(self, logits, temperature=0.8):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.soft_argmax(logits, temperature)
        # y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y


class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2=nn.Parameter(torch.ones(feature))
        self.b_2=nn.Parameter(torch.zeros(feature))
        self.eps=eps
    def forward(self, x):
        mean=x.mean(-1, keepdim=True)
        std=x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

class LookupProb(nn.Module):
    def __init__(self, action2name, config):
        super(LookupProb, self).__init__()
        self.action2name = action2name
        self.use_gpu = config.use_gpu
    def extract_name(self, action_id):
        if str(action_id) in self.action2name.keys():
            action_name = self.action2name[str(action_id)]
            name = torch.Tensor(map(int, action_name.strip().split('-')))
            name = cast_type(name, LONG, self.use_gpu)
            return name
        else:
            return 'empty'
    def forward(self, logits, action_log):
        action_id = torch.argmax(action_log).item()
        name = self.extract_name(action_id)
        if type(name)!=str:
            ids = name.view(-1,1)
            prob = logits.gather(1, ids).view(-1)
            mul_p = 1
            for x in prob:
                mul_p *= x
            return prob.sum()
        else:
            return 0

class HistoryData(nn.Module):
    def __init__(self, maxlen):
        super(HistoryData, self).__init__()
        self.experience_pool = []
        self.maxlen = maxlen

    def add(self, training_batch):
        while (len(self.experience_pool)>=self.maxlen):
            self.experience_pool.pop(random.randrange(len(self.experience_pool)))
        self.experience_pool.append(copy.deepcopy(training_batch))

    def next(self):
        sampled_batch = random.choice(self.experience_pool)
        return sampled_batch
