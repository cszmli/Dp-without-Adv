# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch.autograd import Variable
import torch.nn as nn
from gan_model_sat import Discriminator, Generator, ContEncoder
from torch_utils import one_hot_embedding, LookupProb
from laed.dataset.corpora import PAD, EOS, EOT, BOS
from laed.utils import Pack
from utils import BCELoss_double, cal_accuracy
import torch.nn.functional as F

logger = logging.getLogger()

class GanRnnAgent(nn.Module):
    def __init__(self, corpus, config, action2name):
        super(GanRnnAgent, self).__init__()
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.action2name=action2name
        self.lookupProb_ = LookupProb(action2name, config)
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
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                            self.use_gpu)

    