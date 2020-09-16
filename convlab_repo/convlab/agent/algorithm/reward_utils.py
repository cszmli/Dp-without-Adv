from __future__ import print_function
import numpy as np
import logging
import torch
import os
import json
from collections import defaultdict
INT = 0
LONG = 1
FLOAT = 2
class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


class DataLoader(object):
    logger = logging.getLogger()

    def __init__(self, name, fix_batch=True):
        self.batch_size = 0
        self.ptr = 0
        self.num_batch = None
        self.indexes = None
        self.data_size = None
        self.batch_indexes = None
        self.fix_batch=fix_batch
        self.max_utt_size = None  
        self.name = name

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, *args, **kwargs):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, config, shuffle=True, verbose=True):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // config.batch_size
        if verbose:
            self.logger.info("Number of left over sample %d" % (self.data_size - config.batch_size * self.num_batch))

        # if shuffle and we want to group lines, shuffle batch indexes
        if shuffle and not self.fix_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        if shuffle and self.fix_batch:
            self._shuffle_batch_indexes()

        if verbose:
            self.logger.info("%s begins with %d batches" % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

    def pad_to(self, max_len, tokens, do_pad=True):
        if len(tokens) >= max_len:
            return tokens[0:max_len - 1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

class WoZGanDataLoaders(DataLoader):
    def __init__(self, name, batch_size=16):
        super(WoZGanDataLoaders, self).__init__(name)
        self.action_num = 300
        self.batch_size=batch_size
        data = self._read_file(name)
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data)        
        self.data_size = len(self.data)
        print("Data size: {}".format(self.data_size))
    
    def _read_file(self, dataset):
        # changed part
        # with open(os.path.join('./data/multiwoz', dataset + '.sa.json')) as f:
        # with open(os.path.join('./data/multiwoz', dataset + '.sa_NoHotel.json')) as f: 
        # with open(os.path.join('./data/multiwoz', dataset + '.sa_alldomain.json')) as f:    # ****    
        with open(os.path.join('./data/multiwoz', dataset + '.sa_alldomain_withnext.json')) as f:        
            data = json.load(f)
        return data
     
    def flatten_dialog(self, data):
        results = []
        indexes = []
        batch_indexes = []
        resp_set = set()
        dict_act_seq = defaultdict(list)
        for dlg in data:
            batch_index = []
            state_onehot = dlg['state_onehot']
            state_convlab = dlg['state_convlab']
            action_index = dlg['action']
            action_index_binary = dlg['action_binary']
            state_convlab_next = dlg['state_convlab_next']
            results.append(Pack(action_id=action_index, 
                                state_onehot=state_onehot,
                                action_id_binary=action_index_binary, 
                                state_convlab=state_convlab,
                                state_convlab_next = state_convlab_next
                                ))
            indexes.append(len(indexes))
            batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)
        return results, indexes, batch_indexes

    def epoch_init(self, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        self.num_batch = self.data_size // self.batch_size 
        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
        if verbose:
            print('Number of left over sample = %d' % (self.data_size - self.batch_size * self.num_batch))
        if shuffle:
            if fix_batch:
                self._shuffle_batch_indexes()
            else:
                self._shuffle_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        keys = []
        metas = []
        index = []
        turn_id = []

        state_onehot, state_convlab, state_convlab_next = [], [], []
        action_id, action_id_binary = [], []
        for row in rows:  
            state_onehot.append(row['state_onehot'])
            state_convlab.append(row['state_convlab'])
            action_id.append(row['action_id'])
            action_id_binary.append(row['action_id_binary'])
            state_convlab_next.append(row['state_convlab_next'])
            
        # state = np.array(state)
        # action = np.array(action)
        return Pack(action_id=action_id, 
                    state_onehot=state_onehot,
                    state_convlab=state_convlab, 
                    action_id_binary=action_id_binary,
                    state_convlab_next = state_convlab_next
                    )


def reward_validate(agent, valid_feed):
    with torch.no_grad():
        agent.eval()
        valid_feed.epoch_init(shuffle=False, verbose=True)
        batch_num = 0
        rew_all = 0
        while True:
            batch = valid_feed.next_batch()
            if batch is None:  
                break                                                                                                                     
            rew = agent.forward_validate(batch)
            # wgan_reward.append(torch.stack(acc))
            rew_all += rew.mean().item()
            batch_num+=1
    print("Avg reward: {}".format(rew_all/batch_num))

def one_hot_embedding(labels, num_classes):
    # print(labels)
    if type(labels)==list:
        labels = torch.LongTensor(labels)
    y = torch.eye(num_classes) 
    return y[labels] 