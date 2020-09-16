import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rl.models.model_bases import BaseModel
from laed.dataset.corpora import SYS, EOS, PAD, BOS
from laed.utils import INT, FLOAT, LONG, Pack, cast_type
from laed.enc2dec.encoders import RnnUttEncoder
from laed.enc2dec.decoders import DecoderRNN, GEN, TEACH_FORCE
from laed.criterions import NLLEntropy, CatKLLoss, Entropy, NormKLLoss
from laed import nn_lib
from gan.gan_model import Discriminator, ContEncoder
import numpy as np

class PolicyNN(BaseModel):
    def __init__(self, config):
        super(PolicyNN, self).__init__(config)
        self.p_fc1 = nn.Linear(config.ctx_cell_size, config.ctx_cell_size)
        self.p_y = nn.Linear(config.ctx_cell_size, config.action_num)
    def forward(self, state):
        fc1_out =self.p_fc1(state)
        py_logits = self.p_y(torch.tanh(fc1_out))
        log_py = F.log_softmax(py_logits, dim=1)
        return py_logits, log_py


class LIRL(BaseModel):
    def __init__(self, corpus, config):
        super(LIRL, self).__init__(config)
        self.use_gpu = config.use_gpu
        self.config = config
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.action_number = config.action_num
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        # the hierarchical Context Encoder is pre-trained
        self.ContextEncoder = ContEncoder(corpus, config)
        self.c2z = PolicyNN(config)
        self.z_embedding = nn.Linear(self.action_number, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(self.use_gpu)
       # connector
        self.c_init_connector = nn_lib.LinearConnector(self.action_number,
                                                       config.dec_cell_size,
                                                       config.rnn_cell == 'gru')
               # decoder
        self.embedding = None
        self.decoder = DecoderRNN(self.vocab_size, config.max_dec_len,
                                  config.embed_size, config.dec_cell_size,
                                  self.bos_id, self.eos_id,
                                  n_layers=1, rnn_cell=config.rnn_cell,
                                  input_dropout_p=config.dropout,
                                  dropout_p=config.dropout,
                                  use_attention=config.use_attn,
                                  attn_size=config.dec_cell_size,
                                  attn_mode=config.attn_type,
                                  use_gpu=config.use_gpu,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(torch.log(torch.ones(1) / config.k_size))
        self.eye = Variable(torch.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        # this func can be reused later to add more loss types
        total_loss = loss.nll
        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']
        batch_size = len(ctx_lens)

        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # context encoder
        c_inputs = self.ContextEncoder.utt_encoder(ctx_utts)
        c_outs, c_last = self.ContextEncoder.ctx_encoder(c_inputs, ctx_lens)
        c_last = c_last.squeeze(0)

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        # enc_last = torch.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # DB infor is not fed here
        enc_last = c_last
        logits_py, log_py = self.c2z(enc_last)

        # TODO: modify gumbel_connector to gumble_ST_connector
        sample_y = self.gumbel_connector(logits_py, hard=False)

        # pack attention context
        if self.config.use_attn:
            attn_inputs = c_outs
        else:
            attn_inputs = None
        
        # map sample to initial state of decoder
        # sample_y = sample_y.view(-1, self.action_number)
        # dec_init_state = self.c_init_connector(sample_y) + c_last.unsqueeze(0)
        dec_init_state = self.z_embedding(sample_y.view(-1, self.action_number)) + c_last.unsqueeze(0)
        # dec_init_state = torch.cat([self.z_embedding(sample_y.view(-1, self.action_number)), c_last.unsqueeze(0)], dim=1)
        # decode
        if self.config.rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               inputs=dec_inputs,
                                                               init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_inputs,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            result['nll'] = self.nll(dec_outputs, labels)
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']
        batch_size = len(ctx_lens)

        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)

        # context encoder
        c_inputs = self.utt_encoder(ctx_utts)
        c_outs, c_last = self.ctx_encoder(c_inputs, ctx_lens)
        c_last = c_last.squeeze(0)

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        # enc_last = torch.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # DB infor is not fed here
        enc_last = c_last
        logits_py, log_py = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1) 
        log_qy = F.log_softmax(logits_py, dim=1) 
        idx = torch.multinomial(qy, 1).detach()

        logprob_sample_z = log_qy.gathcher(1, idx).view(-1)
        joint_logpz = torch.sum(logprob_sample_z) 
        sample_y = cast_type(Variable(torch.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = torch.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(torch.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = torch.cat(attn_context, dim=1)
            dec_init_state = torch.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y

