import argparse
import os
import numpy as np
import math
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from laed import nn_lib
from laed.enc2dec import decoders
from laed.enc2dec.decoders import DecoderRNN
from laed.enc2dec.encoders import EncoderRNN, RnnUttEncoder
from laed.dataset.corpora import PAD, EOS, EOT, BOS
from laed.utils import INT, FLOAT, LONG, cast_type, Pack
from gan.torch_utils import GumbelConnector, LayerNorm



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
        total_loss += self.l2_norm()
        total_loss.backward()
        self.clip_gradient()

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

class Generator(BaseModel):
    def __init__(self, config):
        super(Generator, self).__init__(config)
        
        state_in_size = config.state_noise_dim
        action_in_size = config.action_noise_dim
        state_out_size = config.ctx_cell_size
        action_out_size = config.action_num
        


        self.state_model = nn.Sequential(
            # original: 5 block + 1 linear
            nn.Linear(state_in_size, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(config.dropout),

            # nn.Dropout(config.dropout),
            nn.Linear(128, 96),
            # nn.BatchNorm1d(96),
            nn.ReLU(True),
            # nn.Dropout(config.dropout),

            nn.Linear(96, 96),
            # nn.BatchNorm1d(96),
            nn.ReLU(True),
            # nn.Dropout(config.dropout),

            nn.Linear(96, 96),
            # nn.BatchNorm1d(96),
            nn.ReLU(True),


            # nn.Linear(96, 96),
            # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # nn.Linear(96, 96),
            # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # nn.Dropout(config.dropout),
            nn.Linear(96, 128),
            # nn.BatchNorm1d(128),
            # LayerNorm(128),
            nn.ReLU(True),
            # nn.Dropout(config.dropout),

            nn.Linear(128, state_out_size),
        )
        # '''
        self.common_model = nn.Sequential(
            nn.Linear(state_in_size, state_in_size),
            # nn.BatchNorm1d(state_in_size),
            nn.ReLU(True),
            nn.Linear(state_in_size, state_in_size),
            # nn.BatchNorm1d(state_in_size),
            nn.ReLU(True),
            # nn.Linear(state_in_size, 128),
            # # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # # nn.Dropout(config.dropout),

            # # nn.Dropout(config.dropout),
            # nn.Linear(128, 96),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),
            # # nn.Dropout(config.dropout),

            # nn.Linear(96, state_in_size),
            # # nn.BatchNorm1d(state_in_size),
            # nn.ReLU(True),
        )
        # '''
        self.action_model_2 = nn.Sequential(
            # original: 4 block + 1 linear
            nn.Linear(action_in_size, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 96),
            # nn.BatchNorm1d(96),
            nn.ReLU(True),

            nn.Linear(96, 96),
            # nn.BatchNorm1d(96),
            nn.ReLU(True),
            
            # nn.Linear(96, 96),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # nn.Linear(96, 96),
            # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # nn.Linear(96, 96),
            # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            nn.Linear(96, 128),
            # nn.BatchNorm1d(128),
            # LayerNorm(128),
            nn.ReLU(True),

            nn.Linear(128, action_out_size)
        )

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()
        self.clip_gradient()
        
        
        # self.policy_state_norm_layer= LayerNorm(config.ctx_cell_size)
    
    # def get_optimizer(self, config):
        # return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        #    self.parameters()), lr=config.init_lr)

    def forward(self, s_z, a_z):
        
        # no interactions between state and action
        '''
        state_rep = torch.tanh((self.state_model(self.cast_gpu(s_z))))
        action_rep_2 = self.action_model_2(self.cast_gpu(a_z))
        action_rep = self.gumbel_connector.forward_ST(action_rep_2)
        '''

        # state and action share one common MLP
        state_action_pair = self.common_model(self.cast_gpu(s_z))
        # state_rep = torch.tanh(self.state_model(state_action_pair))
        state_rep = torch.sigmoid(self.state_model(state_action_pair))

        action_rep_2 = self.action_model_2(state_action_pair)
        action_rep = self.gumbel_connector.forward_ST(action_rep_2, self.config.gumbel_temp)
        # '''
        return state_rep, action_rep




class Discriminator(BaseModel):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        config.dropout = 0.3
        self.state_in_size = config.ctx_cell_size
        self.action_in_size = config.action_num
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size/2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size/2)
        self.noise_input = 0.01
        self.model = nn.Sequential(
            # nn.BatchNorm1d(self.state_in_size/2 + self.action_in_size/2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            # nn.Dropout(0.5),

            # nn.Linear(self.state_in_size/2, 64),
            nn.Linear(self.state_in_size/2 + self.action_in_size/2, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            # nn.Dropout(0.5),
            # nn.Linear(64, 64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(config.dropout),
            

            # nn.Linear(64, 64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(config.dropout),

            # nn.Linear(64, 64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(config.dropout),

            nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            # nn.Dropout(0.5),

            nn.Linear(32, 1)
        )
        # self.state_nor_layer=LayerNorm(self.state_in_size/2)
        # self.action_nor_layer=LayerNorm(self.action_in_size/2)

    def decay_noise(self):
        self.noise_input *= 0.995

    def forward(self, state, action):
        s_z = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, state.shape))))
        s_a = self.cast_gpu(Variable(torch.Tensor(np.random.normal(0, self.noise_input, action.shape))))
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        # print(state.shape, action_1.shape)
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-6, 1-1e-6)
        return validity
    
    def forward_wgan(self, state, action):
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state_1, action_1], 1)
        validity = self.model(state_action)
        return validity


class ContEncoder(BaseModel):
    def __init__(self, corpus, config):
        super(ContEncoder, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.config = config

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size,
                                      padding_idx=self.rev_vocab[PAD])

        self.utt_encoder = RnnUttEncoder(config.utt_cell_size, 0.0,
                                        bidirection=False,
                                         #  bidirection=True in the original code
                                         use_attn=config.utt_type == 'attn_rnn',
                                         vocab_size=self.vocab_size,
                                         embedding=self.embedding)

        self.ctx_encoder = EncoderRNN(self.utt_encoder.output_size,
                                      config.ctx_cell_size,
                                      0.0,
                                      0.0,
                                      config.num_layer,
                                      config.rnn_cell,
                                      variable_lengths=self.config.fix_batch)


    def forward(self, data_feed):
        ctx_lens = data_feed['context_lens']
        batch_size = len(ctx_lens)

        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        
        # context encoder
        if self.config.hier:
            c_inputs = self.utt_encoder(ctx_utts)
            c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
            c_last = c_last.squeeze(0)
        else:
            c_inputs = self.utt_encoder(ctx_utts)
            c_last = c_inputs.squeeze(1)
        return c_last

class StateEncoder(BaseModel):
    def __init__(self, corpus, config):
        super(StateEncoder, self).__init__(config)
        self.slot_set = a3c_config.slot_set
        self.config = config

    def forward(self, data_feed):
        pass

# this class is to convert the original action to the corresponding latent action
class ActionEncoder(BaseModel):
    def __init__(self, corpus, config, name2action):
        super(ActionEncoder, self).__init__(config)
        self.name2action_dict = name2action
        self.action_num = config.action_num
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        # if self.action_num != len(self.name2action_dict.keys()) + 1:
        #     raise ValueError("the action space should include one spare action to cover some actions \
        #                       that are not in any learned action clusters ")

        self.x_embedding = nn.Embedding(self.vocab_size, config.embed_size)
        self.x_encoder = EncoderRNN(config.embed_size, config.dec_cell_size,
                                    dropout_p=config.dropout,
                                    rnn_cell=config.rnn_cell,
                                    variable_lengths=False)
        self.q_y = nn.Linear(config.dec_cell_size, config.y_size * config.k)
        self.config =config

    def qzx_forward(self, out_utts):
        # this func will be used to extract latent action z given original actions x later in whole pipeline
        output_embedding = self.x_embedding(out_utts)
        x_outs, x_last = self.x_encoder(output_embedding)
        x_last = x_last.transpose(0, 1).contiguous().view(-1, self.config.dec_cell_size)
        qy_logits = self.q_y(x_last).view(-1, self.config.k)
        log_qy = F.log_softmax(qy_logits, dim=1)
        return Pack(qy_logits=qy_logits, log_qy=log_qy)
    
    def forward(self, out_utts):
        out_utts = self.np2var(out_utts, LONG)
        results = self.qzx_forward(self.cast_gpu(out_utts))
        log_qy = results.log_qy.view(-1, self.config.y_size, self.config.k)
        qy = torch.exp(log_qy)
        qy = qy.cpu().data.numpy()
        
        action_list = []
        action_name_list = []
        for b_id in range(out_utts.shape[0]):
            code = []
            for y_id in range(self.config.y_size):
                for k_id in range(self.config.k):
                    if qy[b_id, y_id, k_id] == np.max(qy[b_id, y_id]):
                        code.append(str(k_id))
                        break
            code = '-'.join(code)
            action_id = self.Lookup_action_index(code)
            action_name_list.append(code)
            action_list.append(action_id)
        return action_list, action_name_list

    def Lookup_action_index(self, code):
        if code in self.name2action_dict.keys():
            return self.name2action_dict[code]
        else:
            return self.action_num-1
    



