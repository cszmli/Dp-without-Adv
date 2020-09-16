# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import torch
import os
from collections import defaultdict
import logging
from torch.autograd import Variable
import torch.nn as nn
import gan.gan_model_sat as gan_model_sat
from gan.gan_model import Discriminator, Generator, ContEncoder, ActionEncoder
from gan.torch_utils import one_hot_embedding, LookupProb
from laed.dataset.corpora import PAD, EOS, EOT, BOS, SYS, USR, EOD, BOD, UNK, EOTSELF
from laed.utils import Pack, get_tokenize, get_chat_tokenize, missingdict, FLOAT, LONG, cast_type
from gan.utils import revert_action_dict
import torch.nn.functional as F
import copy
import sys
sys.path.append('./a3c/src')
import a3c.src.deep_dialog.dialog_config as a3c_config

logger = logging.getLogger()

# This class is mainly for preprocessing the raw interactions between user and sys
class DialogExchanger(nn.Module):
    def __init__(self, corpus, config, action2name):
        super(DialogExchanger, self).__init__()
        self.config = config
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.rev_vocab = corpus.rev_vocab
        self.unk_id = corpus.unk_id
        self.vocab_size = len(self.vocab)
        self.tokenize = get_chat_tokenize()
        self.backward_size = config.backward_size
        # self.name2action = revert_action_dict(action2name)
        # actionencoder returns the corresponsing latent action ids given natural languages
        # self.action_encoder = ActionEncoder(corpus, config, self.name2action)

        self.max_utt_size = config.max_utt_len
        self.unk_word = {'PLACEHOLDE'}
        self.gamma = 0.95

    def cast_gpu(self, var):
        if self.config.use_gpu:
            return var.cuda()
        else:
            return var

    def _sent2id_(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]
    
    def _sent2id(self, sent):
        s_l =[]
        for t in sent:
            if t in self.vocab_dict.keys():
                s_l.append(self.vocab_dict[t])
            else:
                s_l.append(self.unk_id)
                self.unk_word.add(t)
        return s_l

    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def extract_slot(line_):
        # print(len(line_))
        if len(line_)<1:
            return 'null'
        line=line_.strip().replace(')', '').split('(')
        out_list = []
        act = line[0]
        contents = line[1]
        content_line = contents.strip().split(';')
        for item in content_line:
            item_list = item.strip().split('=')
            if len(item_list)>1 and (act=='inform' or act=='request'):
                out_list.append('inform'+'_'+item_list[0])
            else:
                out_list.append(act + '_' + item_list[0])
        if len(out_list)==0:
            out_list.append(act)
        out_list=sorted(out_list)
        out_line = ' '.join(out_list)
        return out_line

    def reward_function(self, dialog_status):
        """ Reward Function 1: a reward function based on the dialog_status """
        if dialog_status == -1:
            reward = -0.1 * 40  # 10
        elif dialog_status == 1:
            reward = 0.2 * 40  # 20
        else:
            reward = -0.1
        return reward

    def pad_to(self, max_len, tokens, do_pad=True):
        if len(tokens) >= max_len:
            return tokens[0:max_len - 1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

    def clean_dialog(self, dialogue_list, dialogue_status_list=None, expert_data=False):
        # dialogue_list: len(dialogue_list)==batch_size,  dialogue_list[0]=[{t1}, {t2}, {t3}]
        batch_dialog = []
        for session_id, raw_dlg in enumerate(dialogue_list):
            # norm_dlg = [Pack(speaker=USR, utt=[BOS, BOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size)]
            norm_dlg = []
            state_dict = self.init_state_dict()
            state_dict_feed = [state_dict[k] for k in sorted(state_dict.keys())]
            dialogue_status = dialogue_status_list[session_id]
            for ind, turn_dialog in enumerate(raw_dlg):
                index = str(turn_dialog['dialog_count'])+'-'+str(turn_dialog['turn'])
                # utt = [BOS] + self.tokenize(turn_dialog['Message.Text']) + [EOS]
                utt = [BOS] + self.tokenize(turn_dialog['act_str']) + [EOS]
                turn_id = int(turn_dialog.get('turn', 0))/2 
                reward_ori = self.reward_function(0)
                if ind == len(raw_dlg)-1 and turn_dialog['role']=='agent':
                    reward_ori = self.reward_function(dialogue_status)
                if turn_dialog['role']=='user':
                    norm_dlg.append(Pack(speaker=USR, utt_raw=utt, utt=self._sent2id(utt), index=index, turn_id=turn_id, reward_ori=reward_ori, \
                         raw_act_id=int(turn_dialog.get('action_id', 999)), state_dict=state_dict_feed))
                    if self.config.state_type=='table':
                        state_dict = self.fill_in_state_dict(state_dict, turn_dialog['act_str'], 'user')                    
                        if self.config.input_type=='sat':
                            state_dict_feed = [self.assign_bucket(state_dict[k]) for k in sorted(state_dict.keys())]
                        else:
                            state_dict_feed = [state_dict[k] for k in sorted(state_dict.keys())]                
                else:
                    norm_dlg.append(Pack(speaker=SYS, utt_raw=utt, utt=self._sent2id(utt), index=index, turn_id=turn_id,  reward_ori=reward_ori,\
                        raw_act_id=int(turn_dialog.get('action_id', 0)),state_dict=state_dict_feed))
                    if self.config.state_type=='table':
                        state_dict = self.fill_in_state_dict(state_dict, turn_dialog['act_str'], 'agent')                    
                        if self.config.input_type=='sat':
                            state_dict_feed = [self.assign_bucket(state_dict[k]) for k in sorted(state_dict.keys())]
                        else:
                            state_dict_feed = [state_dict[k] for k in sorted(state_dict.keys())]  
            if norm_dlg[-1]['speaker']==SYS:
                norm_dlg.append(Pack(speaker=USR, utt_raw=[BOS, EOD, EOS], reward_ori=reward_ori, utt=self._sent2id([BOS, EOD, EOS]), index='pad', turn_id=turn_id,\
                    state_dict=state_dict_feed, row_act_id=999))
            batch_dialog.append(Pack(dlg=norm_dlg))

        return batch_dialog
    
    def init_state_dict(self):
        state_dict = defaultdict(int)
        for slot in a3c_config.slot_set_final:
            state_dict[slot]=0
        return state_dict

    def assign_bucket(self, value):
        if 0.9<=value<1.0:
            return 3
        elif 0.7<=value<0.9:
            return 2
        elif 0.1<=value<0.7:
            return 1
        else:
            return 0

 
    def fill_in_state_dict(self, state_dict, utt, role):
        act_list = self.tokenize(utt)
        for word in act_list:
            if word.strip().split('_')[0] in ["confirm_question", "confirm_answer", "thanks", "deny"]:
                word =  word.strip().split('_')[0] + '_'
            entity = role + '_' + word
            if entity in state_dict.keys() and state_dict[entity]==0:
            # if entity in state_dict.keys():
                state_dict[entity] = 1

        for key, value in state_dict.items():
            state_dict[key]=state_dict[key] * self.gamma
        return state_dict
    


    def flatten_dialog(self, dialog_batch):
        # merge dialogue turns in the same batch to one list, using dialogue_border to record the cutting points
        results = []
        dialogue_border = []
        turn_count = 0
        for dlg in dialog_batch:
            for i in range(0, len(dlg.dlg)-1):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - self.backward_size)
                response = dlg.dlg[i].copy()
                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                context = []
                context_temp = []
                if self.config.hier:
                    for turn in dlg.dlg[s_idx: e_idx]:
                        turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                        context.append(turn)
                else:
                    for turn in dlg.dlg[s_idx: e_idx]:
                        context_temp += turn.utt
                        context_temp += [8]
                    turn['utt'] = self.pad_to(self.max_utt_size, context_temp, do_pad=False)
                    context.append(turn)

                
                action_id = dlg.dlg[i].raw_act_id
                turn_id = dlg.dlg[i].turn_id
                state_dict = dlg.dlg[i].state_dict
                reward_ori = dlg.dlg[i].reward_ori
                results.append(Pack(context=context, response=response, index=response.index, turn_id=turn_id, \
                    reward_ori=reward_ori,  action_id=action_id, state_dict=state_dict))
                
                turn_count+=1
            dialogue_border.append(turn_count)
        return results, dialogue_border
    
    def prepare_batch(self, dialog_batch):
        rows = dialog_batch
        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []

        action_target = []
        action_name = []
        index_order = []
        action_id = []
        turn_id_list = []
        state_table = []
        reward_ori_l = []
        for row in rows:
            # print(type(row))
            # print(row)
            in_row, out_row, index, a_id, turn_id = row.context, row.response, row.index, row.action_id, row.turn_id
            index_order.append(index)
            # the contexts for one turn in a dialogue
            batch_ctx = []
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_size, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)
            ctx_lens.append(len(batch_ctx))

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            action_id.append(a_id)
            turn_id_list.append(turn_id)
            state_table.append(row.state_dict)
            reward_ori_l.append(row.reward_ori)

        batch_size = len(ctx_lens)

        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((batch_size, max_ctx_len, self.max_utt_size), dtype=np.int32)

        vec_out_lens = np.array(out_lens)  # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((batch_size, max_out_len), dtype=np.int32)

        for b_id in range(batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]

        # action_id, action_name = self.action_encoder(vec_out_utts)

        return Pack(context_lens=vec_ctx_lens, # (batch_size, )
                    contexts=vec_ctx_utts, # (batch_size, max_ctx_len, max_utt_len)
                    output_lens=vec_out_lens, # (batch_size, )
                    outputs=vec_out_utts, # (batch_size, max_out_len)
                    action_id=action_id,
                    turn_id=turn_id_list,
                    index_rec=index_order,
                    state_table=state_table,
                    reward_ori=reward_ori_l
                    )

    def forward(self, dialogue_list, dialogue_status_list=None):
        # dialogue_list: len(dialogue_list)==batch_size,  dialogue_list[0]=[{t1}, {t2}, {t3}]
        batch_dialog = self.clean_dialog(dialogue_list, dialogue_status_list)
        batch_dialog, dialog_cut = self.flatten_dialog(batch_dialog)
        batch_dialog = self.prepare_batch(batch_dialog)
        return batch_dialog, dialog_cut

         


class RewardAgent(nn.Module):
    def __init__(self, corpus, config, action2name=None):
        super(RewardAgent, self).__init__()
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.go_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.context_encoder = ContEncoder(corpus, config)
        self.discriminator = Discriminator(config)
        self.data_exchanger = DialogExchanger(corpus, config, action2name)
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
                         self.config.use_gpu)
        return cast_type(Variable(torch.from_numpy(inputs)), dtype,
                         self.config.use_gpu)

    def read_context(self, batch_feed):
        '''
        batch_feed:  {'action_id':?, 'context_lens':?, 'contexts':?}
        '''
        action_data_feed = one_hot_embedding(batch_feed['action_id'], self.config.action_num)
        if self.config.state_type=='rnn':
            real_state_rep = self.context_encoder(batch_feed).detach()
        elif self.config.state_type=='table':
            if self.config.input_type=='sat':
                real_state_rep = self.np2var(batch_feed['state_table'], LONG)
                real_state_rep = one_hot_embedding(real_state_rep.view(-1), 3).view(-1, self.state_out_size)
            else:
                real_state_rep = self.np2var(batch_feed['state_table'], FLOAT)
        return real_state_rep, action_data_feed

    def reward_step(self, batch_feed):
        '''
        batch_feed:  {'action_id':?, 'context_lens':?, 'contexts':?}
        '''
        real_state_rep, action_data_feed = self.read_context(batch_feed)
        disc_v = self.discriminator(real_state_rep, action_data_feed)
        # disc_v = -torch.log(1-disc_v)
        return disc_v.detach(), real_state_rep.detach()
    
    def reorder_reward(self, reward, dialog_cut):
        reward_mat = []
        if len(reward)!=dialog_cut[-1]:
            print(len(reward), dialog_cut)
            raise ValueError("the reward number and system response times should be equal")
        for i in range(0, len(dialog_cut)-1):
            st_idx = 0 if i==0 else dialog_cut[i-1]
            ed_idx = dialog_cut[i]
            reward_mat.append(reward[st_idx:ed_idx])
        return reward_mat

    
    def pairing_reward(self, reward, reward_index, state_rep, action_index, reward_ori_index):
        reward_pair = defaultdict(int)
        state_rep_pair = defaultdict(int)
        action_pair = defaultdict(int)
        reward_ori_pair = defaultdict(float)
        for ind, reward_id in enumerate(reward_index):
            reward_pair[reward_id] = reward[ind]
            # concatenate the next state 
            if ind<len(reward_index)-1:
                index_diff = int(reward_id.strip().split('-')[-1]) - int(reward_index[ind+1].strip().split('-')[-1])
            else:
                index_diff = 0
            state_rep_pair[reward_id] = (state_rep[ind], state_rep[ind+1]) if index_diff==-2 else (state_rep[ind], state_rep[ind])
            action_pair[reward_id] = action_index[ind]
            reward_ori_pair[reward_id] = reward_ori_index[ind]
        # for reward_id, reward_v in zip(reward_index, reward):
            # reward_pair[reward_id] = reward_v
        return reward_pair, state_rep_pair, action_pair, reward_ori_pair

    
    def forward(self, dialogue_list, dialogue_status_list=None):
        # dialogue_list: len(dialogue_list)==batch_size,  dialogue_list[0]=[{t1}, {t2}, {t3}]
        # t1 = {role:'user', act_str:'inform_numberofpeople inform_movie_name'}
        # t2 = {role:'sys', act_str:'request_theater request_date'}
        batch_dialog, dialog_cut = self.data_exchanger(dialogue_list, dialogue_status_list)
        reward_index = copy.deepcopy(batch_dialog.index_rec)
        action_index = copy.deepcopy(batch_dialog.action_id)
        reward_ori_index = copy.deepcopy(batch_dialog.reward_ori)
        reward, reward_state_rep = self.reward_step(batch_dialog)
        # reward_final = self.reorder_reward(reward, dialog_cut)
        reward_final, state_rep_pair, action_pair, reward_ori_pair = self.pairing_reward(reward.view(-1).cpu().tolist(), reward_index, reward_state_rep.cpu(), action_index, reward_ori_index)
        return reward_final, state_rep_pair, action_pair, reward_ori_pair

    def fetch_state_rep(self, dialogue_list, dialogue_status_list=None):
        batch_dialog, _ = self.data_exchanger(dialogue_list, dialogue_status_list)
        real_state_rep, _ = self.read_context(batch_dialog)
        return real_state_rep[-1].unsqueeze(0).cpu().detach()



class RewardAgent_SAT(RewardAgent):
    def __init__(self, corpus, config, action2name=None):
        super(RewardAgent_SAT, self).__init__(corpus, config, action2name=None)
        self.discriminator = gan_model_sat.Discriminator(config)
        self.state_out_size = config.ctx_cell_size * config.bucket_num
        self.config = config    

    def read_context(self, batch_feed):
        '''
        batch_feed:  {'action_id':?, 'context_lens':?, 'contexts':?}
        '''
        action_data_feed = one_hot_embedding(batch_feed['action_id'], self.config.action_num)
        if self.config.state_type=='rnn':
            real_state_rep = self.context_encoder(batch_feed).detach()
        elif self.config.state_type=='table':
            if self.config.input_type=='sat':
                real_state_rep = self.np2var(batch_feed['state_table'], LONG)
                real_state_rep = one_hot_embedding(real_state_rep.view(-1), self.config.bucket_num).view(-1, self.state_out_size)
            else:
                real_state_rep = self.np2var(batch_feed['state_table'], FLOAT)
        return real_state_rep, action_data_feed
