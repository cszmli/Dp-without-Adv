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

class Generator(BaseModel):
    def __init__(self, config):
        super(Generator, self).__init__(config)
        
        state_in_size = config.state_noise_dim
        action_in_size = config.action_noise_dim
        turn_in_size = config.action_noise_dim
        self.state_out_size = config.ctx_cell_size 
        self.state_out_size_final = config.ctx_cell_size * config.bucket_num
        action_out_size = config.action_num
        
        self.gumble_num = config.ctx_cell_size
        self.gumble_length_index = [config.bucket_num] * self.gumble_num 
        self.last_layers = nn.ModuleList()
        for gumbel_width in self.gumble_length_index:
            self.last_layers.append(nn.Linear(self.state_out_size, gumbel_width))
        

        self.state_model = nn.Sequential(
            # original: 5 block + 1 linear
            nn.Linear(state_in_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            
            nn.Linear(128, 96),
            # nn.BatchNorm1d(96),
            nn.ReLU(True),

            nn.Linear(96, 96),
            # nn.BatchNorm1d(96),
            nn.ReLU(True),

            # nn.Dropout(config.dropout),
            nn.Linear(96, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # nn.Dropout(config.dropout),

            nn.Linear(128, self.state_out_size),
        )
        # '''
        self.common_model = nn.Sequential(
            nn.Linear(state_in_size, state_in_size),
            nn.BatchNorm1d(state_in_size),
            nn.ReLU(True),
            
            nn.Linear(state_in_size, 96),
            # nn.BatchNorm1d(96),
            nn.ReLU(True),

            # # nn.Dropout(config.dropout),
            # nn.Linear(96, 96),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # nn.Linear(96, 96),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),
            # # nn.Dropout(config.dropout),

            nn.Linear(96, state_in_size),
            nn.BatchNorm1d(state_in_size),
            nn.ReLU(True),
        )
        # '''
        self.action_model_2 = nn.Sequential(
            # original: 4 block + 1 linear
            # nn.Linear( self.state_out_size, 128),
            nn.Linear(action_in_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            
             nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(True),

            nn.Linear(96, action_out_size)
        )
    def gumble_index_multiwoz(self):
        index = 4  * [6] + 3 * [2] + 60 * [2] + 204 * [2] + 3 * [2] + 2 * [3] + 1 * [2] + 4 * [2] + 4 * [3] \
                 + 1 * [2] + 1 * [2] + 1 * [3] + 1 * [2] + 4 * [2] + 7 * [3] + 1 * [2] + 1 * [2] + 4 * [3] + 1 * [2] \
                 + 3 * [2] + 5 * [3] + 1 * [2] + 1 * [2] + 1 * [2] + 1 * [5]

    def backward(self, batch_cnt, loss):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()
        self.clip_gradient()


    def forward_(self, s_z):
        # state and action share one common MLP
        state_action_turn_pair = self.common_model(self.cast_gpu(s_z))
        # state_rep = torch.tanh(self.state_model(state_action_turn_pair))
        state_rep = self.state_model(state_action_turn_pair)
        state_rep = self.gumbel_connector.forward_ST(state_rep.view(-1,  self.config.bucket_num), self.config.gumbel_temp)
        state_rep = state_rep.view(-1,self.state_out_size )
        action_rep_2 = self.action_model_2(state_action_turn_pair)
        # action_rep_2 = self.action_model_2(state_rep)
        action_rep = self.gumbel_connector.forward_ST(action_rep_2, self.config.gumbel_temp)
        return state_rep, action_rep
    
    def forward(self, s_z):
        # state and action share one common MLP
        state_action_turn_pair = self.common_model(self.cast_gpu(s_z))
        # state_rep = torch.tanh(self.state_model(state_action_turn_pair))
        state_rep = self.state_model(state_action_turn_pair)
        input_to_gumble = []
        for layer, g_width in zip(self.last_layers, self.gumble_length_index):
            out = layer(state_rep)
            out = self.gumbel_connector.forward_ST(out.view(-1,  g_width), self.config.gumbel_temp)
            input_to_gumble.append(out)
        state_rep = torch.cat(input_to_gumble, -1)
        # print(state_rep.size())
        state_rep = state_rep.view(-1,self.state_out_size_final )
        action_rep_2 = self.action_model_2(state_action_turn_pair)
        # action_rep_2 = self.action_model_2(state_rep)
        action_rep = self.gumbel_connector.forward_ST(action_rep_2, self.config.gumbel_temp)
        return state_rep, action_rep




class Discriminator(BaseModel):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.state_in_size = config.ctx_cell_size * config.bucket_num
        self.action_in_size = config.action_num
        # self.state_rep = nn.Linear(self.state_in_size * 2, self.state_in_size/2)
        # self.action_rep = nn.Linear(self.action_in_size * 2, self.action_in_size/2)
        self.state_rep = nn.Linear(self.state_in_size, self.state_in_size/2)
        self.action_rep = nn.Linear(self.action_in_size, self.action_in_size/2)
        
        self.noise_input = 0.01
        self.model = nn.Sequential(
            # nn.BatchNorm1d(self.state_in_size/2 + self.action_in_size/2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Tanh(),
            nn.Dropout(config.dropout),
            # nn.Dropout(0.5),

            # nn.Linear(self.state_in_size/2, 64),
            nn.Linear(self.state_in_size/2 + self.action_in_size/2, 64),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Tanh(),
            
            nn.Dropout(config.dropout),
            # nn.Dropout(0.5),

            nn.Linear(64, 1),
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            # # nn.Tanh(),        
            # nn.Dropout(config.dropout),
            # # nn.Dropout(0.5),
            # nn.Linear(32, 1)
        )
        # self.state_nor_layer=LayerNorm(self.state_in_size/2)
        # self.action_nor_layer=LayerNorm(self.action_in_size/2)

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
        state_1 = self.state_rep(self.cast_gpu(state))
        action_1 = self.action_rep(self.cast_gpu(action))
        state_action = torch.cat([state_1, action_1], 1)
        validity = torch.sigmoid(self.model(state_action))
        validity = torch.clamp(validity, 1e-7, 1-1e-7)
        return validity
    

class ContEncoder(BaseModel):
    def __init__(self, corpus, config):
        super(ContEncoder, self).__init__(config)
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]

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
        # out_utts = self.np2var(data_feed['outputs'], LONG)
        # prev_utts = self.np2var(data_feed['prevs'], LONG)
        # next_utts = self.np2var(data_feed['nexts'], LONG)
        # action_id = self.np2var(data_feed['action_id'], LONG)

        # context encoder
        c_inputs = self.utt_encoder(ctx_utts)
        # c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
        # c_last = c_last.squeeze(0)
        c_last = c_inputs.squeeze(1)
        return c_last
        
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
<<<<<<< HEAD
            nn.ReLU(True),

            # nn.Dropout(config.dropout),
            nn.Linear(100, 128),
=======
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            
            # nn.Linear(100, 100),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # nn.Linear(100, 100),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # nn.Dropout(config.dropout),
            nn.Linear(100, 128),
            nn.BatchNorm1d(128),
>>>>>>> d6f985ec03a61d340f5c55d840b3e8573cf13535
            nn.ReLU(True),
            # nn.Dropout(config.dropout),

            nn.Linear(128, self.state_out_size),
        )
        # '''
        self.common_model = nn.Sequential(
            nn.Linear(state_in_size, state_in_size),
            nn.ReLU(True),
            
            # nn.Linear(state_in_size, 96),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # # nn.Dropout(config.dropout),
            # nn.Linear(96, 96),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            # nn.Linear(96, 96),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),
            # # nn.Dropout(config.dropout),

<<<<<<< HEAD
            # nn.Linear(state_in_size, state_in_size),
            # # nn.BatchNorm1d(state_in_size),
            # nn.ReLU(True),
=======
            nn.Linear(96, state_in_size),
            # nn.BatchNorm1d(state_in_size),
            nn.ReLU(True),
>>>>>>> d6f985ec03a61d340f5c55d840b3e8573cf13535
        )
        # '''
        self.action_model_2 = nn.Sequential(
            # nn.Linear( self.state_out_size_final, 96),
            nn.Linear( self.state_out_size, 96),            
<<<<<<< HEAD
            nn.ReLU(True),
            
            nn.Linear(96, 96),
=======
            # nn.Linear(action_in_size, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(True),
            
            # nn.Linear(96, 96),
            # nn.ReLU(True),
            
             nn.Linear(96, 96),
            nn.BatchNorm1d(96),
>>>>>>> d6f985ec03a61d340f5c55d840b3e8573cf13535
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
<<<<<<< HEAD
            out = self.gumbel_connector.forward_ST_soft(out.view(-1,  g_width), self.config.gumbel_temp)
=======
            out = self.gumbel_connector.forward_ST(out.view(-1,  g_width), self.config.gumbel_temp)
>>>>>>> d6f985ec03a61d340f5c55d840b3e8573cf13535
            input_to_gumble.append(out)
        state_rep = torch.cat(input_to_gumble, -1)
        # print(state_rep.size())
        state_rep = state_rep.view(-1,self.state_out_size_final )
        # action_rep_2 = self.action_model_2(state_action_turn_pair)
        # action_rep_2 = self.action_model_2(state_rep)
        action_rep_2 = self.action_model_2(state_rep1)        
<<<<<<< HEAD
        action_rep = self.gumbel_connector.forward_ST_soft(action_rep_2, self.config.gumbel_temp)
=======
        action_rep = self.gumbel_connector.forward_ST(action_rep_2, self.config.gumbel_temp)
>>>>>>> d6f985ec03a61d340f5c55d840b3e8573cf13535
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
            # nn.BatchNorm1d(100),
            nn.ReLU(True),
            
            # nn.Linear(100, 100),
            # # nn.BatchNorm1d(96),
            # nn.ReLU(True),

            nn.Linear(100, self.state_out_size),
            # nn.BatchNorm1d(self.state_out_size),
            nn.ReLU(True),

            nn.Linear(self.state_out_size, self.state_out_size),
        )
        # '''
        self.common_model = nn.Sequential(
            nn.Linear(state_in_size, state_in_size),
            # nn.BatchNorm1d(state_in_size),
            nn.ReLU(True),
            
            nn.Linear(state_in_size, state_in_size),
            # nn.BatchNorm1d(state_in_size),
            nn.ReLU(True),

        )
        # self.action_model_2 = nn.Sequential(
        #     # nn.Linear( self.state_out_size, 48),            
        #     nn.Linear(state_in_size, 48),
        #     # nn.BatchNorm1d(48),
        #     nn.ReLU(True),
            
        #     nn.Linear(48, 48),
        #     # nn.BatchNorm1d(48),
        #     nn.ReLU(True),

        #     nn.Linear(48, self.action_out_size)
        # )

        self.action_model_2 = nn.Sequential(
            # nn.Linear( self.state_out_size, 96),            
            nn.Linear(state_in_size, 96),
            # nn.BatchNorm1d(48),
            nn.ReLU(True),
            
            nn.Linear(96, 96),
            # nn.BatchNorm1d(48),
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

            # nn.Linear(self.state_in_size/2, 64),
            nn.Linear(self.state_in_size/2 + self.action_in_size/3, 32),
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            # nn.Dropout(0.5),
            
<<<<<<< HEAD
            nn.Linear(32, 32),
=======
            nn.Linear(64, 32),
>>>>>>> d6f985ec03a61d340f5c55d840b3e8573cf13535
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),

            nn.Linear(32, 1),
            # nn.BatchNorm1d(32),
            # nn.LeakyReLU(0.2, inplace=True),
            # # nn.Tanh(),        
            # nn.Dropout(config.dropout),
            # # nn.Dropout(0.5),
            # nn.Linear(32, 1)
        )
        # self.state_nor_layer=LayerNorm(self.state_in_size/2)
        # self.action_nor_layer=LayerNorm(self.action_in_size/2)

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
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            # nn.Dropout(0.5),
            
            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
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
            # nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            # nn.Dropout(0.5),
            
            nn.Linear(32, 32),
            # nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )
