#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University
from __future__ import print_function
import numpy as np
from laed.utils import Pack
from laed.dataset.dataloader_bases import DataLoader
from laed.dataset.corpora import USR, SYS, BOT, EOT
import json
import os
import os.path as path
from collections import defaultdict

# Stanford Multi Domain
class SMDDataLoader(DataLoader):
    def __init__(self, name, data, config):
        super(SMDDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = range(len(self.data))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)):
                e_id = i
                s_id = max(0, e_id - backward_size)
                response = dialog[i].copy()
                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)
                results.append(Pack(context=contexts, response=response))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        # input_context, context_lens, floors, topics, a_profiles, b_Profiles, outputs, output_lens
        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas)


class SMDDialogSkipLoader(DataLoader):
    def __init__(self, name, data, config):
        super(SMDDialogSkipLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = range(len(self.data))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size, response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens)


# Daily Dialog
class DailyDialogSkipLoader(DataLoader):
    def __init__(self, name, data, config):
        super(DailyDialogSkipLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.name = name
        self.max_utt_size = config.max_utt_len
        self.data = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        if config.fix_batch:
            all_ctx_lens = [len(d.context) for d in self.data]
            self.indexes = list(np.argsort(all_ctx_lens))[::-1]
        else:
            self.indexes = range(len(self.data))

    def flatten_dialog(self, data, backward_size):
        results = []
        for dialog in data:
            for i in range(1, len(dialog)-1):
                e_id = i
                s_id = max(0, e_id - backward_size)

                response = dialog[i]
                prev = dialog[i - 1]
                next = dialog[i + 1]

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                contexts = []
                for turn in dialog[s_id:e_id]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    contexts.append(turn)

                results.append(Pack(context=contexts, response=response,
                                    prev_resp=prev, next_resp=next))
        return results

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        context_lens, context_utts, out_utts, out_lens = [], [], [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []
        metas = []
        for row in rows:
            ctx = row.context
            resp = row.response

            out_utt = resp.utt
            context_lens.append(len(ctx))
            context_utts.append([turn.utt for turn in ctx])

            out_utt = out_utt
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            metas.append(resp.meta)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))

        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens),
                                self.max_utt_size), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]

            # fill the context tensor
            new_array = np.empty((vec_context_lens[b_id], self.max_utt_size))
            new_array.fill(0)
            for i, row in enumerate(context_utts[b_id]):
                for j, ele in enumerate(row):
                    new_array[i, j] = ele
            vec_context[b_id, 0:vec_context_lens[b_id], :] = new_array

        return Pack(contexts=vec_context, context_lens=vec_context_lens,
                    outputs=vec_outs, output_lens=vec_out_lens,
                    metas=metas, prevs=vec_prevs, prev_lens=vec_prev_lens,
                    nexts=vec_nexts, next_lens=vec_next_lens)


# PTB
class PTBDataLoader(DataLoader):

    def __init__(self, name, data, config):
        super(PTBDataLoader, self).__init__(name, fix_batch=config.fix_batch)
        self.max_utt_size = config.max_utt_len
        self.data = self.pad_data(data)
        self.data_size = len(self.data)
        all_lens = [len(line.utt) for line in self.data]
        print("Max len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                            np.min(all_lens),
                                                            float(np.mean(all_lens))))
        if config.fix_batch:
            self.indexes = list(np.argsort(all_lens))
        else:
            self.indexes = range(len(self.data))

    def pad_data(self, data):
        for l in data:
            l['utt'] = self.pad_to(self.max_utt_size, l.utt, do_pad=False)
        return data

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        input_lens = np.array([len(row.utt) for row in rows], dtype=np.int32)
        max_len = np.max(input_lens)
        inputs = np.zeros((self.batch_size, max_len), dtype=np.int32)
        for idx, row in enumerate(rows):
            inputs[idx, 0:input_lens[idx]] = row.utt

        return Pack(outputs=inputs, output_lens=input_lens, metas=[None]*len(rows))


class BeliefDbDataLoaders(DataLoader):
    def __init__(self, name, data, config):
        super(BeliefDbDataLoaders, self).__init__(name)
        self.max_utt_size = config.max_utt_len
        self.dialog_no_id = 0
        self.action_num = config.action_num
        self.action_index = self.load_action_index(name, config)
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        print("Data size: {}".format(self.data_size))
        self.domains = ['hotel', 'restaurant', 'train', 'attraction', 'hospital', 'police', 'taxi']
        
        

    def load_action_index(self,name, config):
        action_file = os.path.join(config.log_dir, config.action_id_path + '.{}'.format(name))
        if not os.path.isfile(action_file):
            return None
        else:
            with open(action_file, 'r') as f:
                action_data = json.load(f)
            return action_data

    def flatten_dialog(self, data, backward_size):
        results = []
        indexes = []
        batch_indexes = []
        resp_set = set()

        for dlg in data:
            goal = dlg.goal
            key = dlg.key
            batch_index = []
            for i in range(1, len(dlg.dlg)-1):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - backward_size)
                response = dlg.dlg[i].copy()
                prev = dlg.dlg[i - 1].copy()
                next = dlg.dlg[i + 1].copy()

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                resp_set.add(json.dumps(response.utt))
                context = []
                for turn in dlg.dlg[s_idx: e_idx]:
                    turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                    context.append(turn)
                if self.action_index is not None and str(len(indexes)) in self.action_index and type(self.action_index[str(len(indexes))][1])==int:
                    action_id = self.action_index[str(len(indexes))][1]
                    action_id=int(action_id)
                    action_seq = self.action_index[str(len(indexes))][0]
                else:
                    action_id = self.action_num - 1
                    action_seq = 'empty'
                    self.dialog_no_id += 1
                results.append(Pack(context=context, response=response, goal=goal,  prev_resp=prev, next_resp=next, 
                                     key=key, raw_index=len(indexes), action_id=action_id, action_seq=action_seq))
                indexes.append(len(indexes))
                batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)
            # results.append(Pack(context=contexts, response=response,
                                    # prev_resp=prev, next_resp=next))

        print("Unique resp {}".format(len(resp_set)))
        print("Dialogues without ID: {}".format(self.dialog_no_id))
        return results, indexes, batch_indexes

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        if fix_batch:
            self.batch_size = None
            self.num_batch = len(self.batch_indexes)
        else:
            self.batch_size = config.batch_size
            self.num_batch = self.data_size // config.batch_size
            self.batch_indexes = []
            for i in range(self.num_batch):
                self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
            if verbose:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
        if shuffle:
            if fix_batch:
                self._shuffle_batch_indexes()
            else:
                self._shuffle_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []

        out_bs, out_db = [] , []
        goals, goal_lens = [], [[] for _ in range(len(self.domains))]
        keys = []
        metas = []
        index = []
        action_target = []
        action_name = []
        for row in rows:
            # print(type(row))
            # print(row)
            in_row, out_row, goal_row, index_row = row.context, row.response, row.goal, row.raw_index
            
            # source context
            keys.append(row.key)
            index.append(index_row)
            action_target.append(row.action_id)
            # results = map(int, results)
            if row.action_seq=='empty':
                action_name.append('empty')
            else:
                action_name.append(map(int, row.action_seq.strip().split('-')))
            # action_name.append(map(int, row.action_seq.strip().split('-')))

            batch_ctx = []
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_size, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)
            ctx_lens.append(len(batch_ctx))

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            metas.append('meta')
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            out_bs.append(out_row.bs)
            out_db.append(out_row.db)

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))
            # goal
            goals.append(goal_row)
            for i, d in enumerate(self.domains):
                goal_lens[i].append(len(goal_row[d]))

        batch_size = len(ctx_lens)

        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((batch_size, max_ctx_len, self.max_utt_size), dtype=np.int32)

        vec_out_bs = np.array(out_bs) # (batch_size, 94)
        vec_out_db = np.array(out_db) # (batch_size, 30)

        vec_out_lens = np.array(out_lens)  # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((batch_size, max_out_len), dtype=np.int32)

        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        max_goal_lens, min_goal_lens = [max(ls) for ls in goal_lens], [min(ls) for ls in goal_lens]
        if max_goal_lens != min_goal_lens:
            print('Fatal Error!')
            exit(-1)
        self.goal_lens = max_goal_lens
        vec_goals_list = [np.zeros((batch_size, l), dtype=np.float32) for l in self.goal_lens]

        for b_id in range(batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]
            for i, d in enumerate(self.domains):
                vec_goals_list[i][b_id, :] = goals[b_id][d]

        return Pack(context_lens=vec_ctx_lens, # (batch_size, )
                    contexts=vec_ctx_utts, # (batch_size, max_ctx_len, max_utt_len)
                    output_lens=vec_out_lens, # (batch_size, )
                    outputs=vec_out_utts, # (batch_size, max_out_len)
                    metas=metas,
                    bs=vec_out_bs, # (batch_size, 94)
                    db=vec_out_db, # (batch_size, 30)
                    prevs=vec_prevs, 
                    prev_lens=vec_prev_lens,
                    nexts=vec_nexts, 
                    next_lens=vec_next_lens,
                    goals_list=vec_goals_list, # 7*(batch_size, bow_len), bow_len differs w.r.t. domain
                    keys=keys,
                    index=index,
                    action_id=action_target,
                    action_name=action_name)



class MovieDataLoaders(DataLoader):
    def __init__(self, name, data, config):
        super(MovieDataLoaders, self).__init__(name)
        self.max_utt_size = config.max_utt_len
        self.dialog_no_id = 0
        self.action_num = config.action_num
        self.action_index = self.load_action_index(name, config)
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        print("Data size: {}".format(self.data_size))
        
        

    def load_action_index(self,name, config):
        action_file = os.path.join(config.log_dir, config.action_id_path + '.{}'.format(name))
        if not os.path.isfile(action_file):
            return None
        else:
            with open(action_file, 'r') as f:
                action_data = json.load(f)
            return action_data
    

    def flatten_dialog(self, data, backward_size):
        results = []
        indexes = []
        batch_indexes = []
        resp_set = set()
        dict_act_seq = defaultdict(list)
        for dlg in data:
            key = dlg.key
            batch_index = []
            for i in range(1, len(dlg.dlg)-1):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - backward_size)
                response = dlg.dlg[i].copy()
                prev = dlg.dlg[i - 1].copy()
                next = dlg.dlg[i + 1].copy()

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                resp_set.add(json.dumps(response.utt))
                context = []
                # for turn in dlg.dlg[s_idx: e_idx]:
                #     turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                #     context.append(turn)

                contx_temp = []
                for turn in dlg.dlg[s_idx: e_idx]:
                    contx_temp += turn.utt
                    contx_temp += [8]
                turn['utt'] = self.pad_to(self.max_utt_size, contx_temp, do_pad=False)
                context.append(turn)
                if self.action_index is not None and str(len(indexes)) in self.action_index and type(self.action_index[str(len(indexes))][1])==int:
                    action_id = self.action_index[str(len(indexes))][1]
                    action_id=int(action_id)
                    action_seq = self.action_index[str(len(indexes))][0]
                    dict_act_seq[action_id]=action_seq
                else:
                    action_id = self.action_num - 1
                    action_seq = 'empty'
                    self.dialog_no_id += 1
                    dict_act_seq[action_id]=action_seq
                results.append(Pack(context=context, response=response, prev_resp=prev, next_resp=next, 
                                     key=key, raw_index=len(indexes), action_id=action_id, action_seq=action_seq))
                indexes.append(len(indexes))
                batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)
            # results.append(Pack(context=contexts, response=response,
                                    # prev_resp=prev, next_resp=next))

        print("Unique resp {}".format(len(resp_set)))
        print("Dialogues without ID: {}".format(self.dialog_no_id))
        return results, indexes, batch_indexes

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        if fix_batch:
            self.batch_size = None
            self.num_batch = len(self.batch_indexes)
        else:
            self.batch_size = config.batch_size
            self.num_batch = self.data_size // config.batch_size
            self.batch_indexes = []
            for i in range(self.num_batch):
                self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
            if verbose:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
        if shuffle:
            if fix_batch:
                self._shuffle_batch_indexes()
            else:
                self._shuffle_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []

        out_bs, out_db = [] , []
        keys = []
        metas = []
        index = []
        action_target = []
        action_name = []
        for row in rows:
            # print(type(row))
            # print(row)
            in_row, out_row, index_row = row.context, row.response, row.raw_index
            
            # source context
            keys.append(row.key)
            index.append(index_row)
            action_target.append(row.action_id)
            # results = map(int, results)
            if row.action_seq=='empty':
                action_name.append('empty')
            else:
                action_name.append(map(int, row.action_seq.strip().split('-')))
            # action_name.append(map(int, row.action_seq.strip().split('-')))

            batch_ctx = []
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_size, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)
            ctx_lens.append(len(batch_ctx))

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            metas.append('meta')
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))
        batch_size = len(ctx_lens)

        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((batch_size, max_ctx_len, self.max_utt_size), dtype=np.int32)

        vec_out_lens = np.array(out_lens)  # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((batch_size, max_out_len), dtype=np.int32)

        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]


        return Pack(context_lens=vec_ctx_lens, # (batch_size, )
                    contexts=vec_ctx_utts, # (batch_size, max_ctx_len, max_utt_len)
                    output_lens=vec_out_lens, # (batch_size, )
                    outputs=vec_out_utts, # (batch_size, max_out_len)
                    metas=metas,
                    prevs=vec_prevs, 
                    prev_lens=vec_prev_lens,
                    nexts=vec_nexts, 
                    next_lens=vec_next_lens,
                    keys=keys,
                    index=index,
                    action_id=action_target,
                    action_name=action_name)



class ArtDataLoaders(DataLoader):
    def __init__(self, name, data, config):
        super(ArtDataLoaders, self).__init__(name)
        self.config = config
        self.max_utt_size = config.max_utt_len
        self.dialog_no_id = 0
        self.action_num = config.action_num
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data, config.backward_size)
        self.data_size = len(self.data)
        print("Data size: {}".format(self.data_size))
        
    def flatten_dialog(self, data, backward_size):
        results = []
        indexes = []
        batch_indexes = []
        resp_set = set()
        dict_act_seq = defaultdict(list)
        for dlg in data:
            key = dlg.key
            batch_index = []
            for i in range(1, len(dlg.dlg)-1):
                if dlg.dlg[i].speaker == USR:
                    continue
                e_idx = i
                s_idx = max(0, e_idx - backward_size)
                response = dlg.dlg[i].copy()
                prev = dlg.dlg[i - 1].copy()
                next = dlg.dlg[i + 1].copy()

                response['utt'] = self.pad_to(self.max_utt_size,response.utt, do_pad=False)
                prev['utt'] = self.pad_to(self.max_utt_size, prev.utt, do_pad=False)
                next['utt'] = self.pad_to(self.max_utt_size, next.utt, do_pad=False)

                resp_set.add(json.dumps(response.utt))
                context = []
                
                if self.config.hier:
                    for turn in dlg.dlg[s_idx: e_idx]:
                        turn['utt'] = self.pad_to(self.max_utt_size, turn.utt, do_pad=False)
                        context.append(turn)
                else:
                    contx_temp = []
                    for turn in dlg.dlg[s_idx: e_idx]:
                        contx_temp += turn.utt
                        contx_temp += [8]
                    turn['utt'] = self.pad_to(self.max_utt_size, contx_temp, do_pad=False)
                    context.append(turn)

                action_id = dlg.dlg[i].raw_act_id
                turn_id = dlg.dlg[i].turn_num
                state_dict = dlg.dlg[i].state_dict
                results.append(Pack(context=context, response=response, prev_resp=prev, next_resp=next, 
                                     key=key, raw_index=len(indexes), action_id=action_id, turn_id=turn_id, state_dict=state_dict))
                indexes.append(len(indexes))
                batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)
            # results.append(Pack(context=contexts, response=response,
                                    # prev_resp=prev, next_resp=next))

        print("Unique resp {}".format(len(resp_set)))
        return results, indexes, batch_indexes

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        if fix_batch:
            self.batch_size = None
            self.num_batch = len(self.batch_indexes)
        else:
            self.batch_size = config.batch_size
            self.num_batch = self.data_size // config.batch_size
            self.batch_indexes = []
            for i in range(self.num_batch):
                self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
            if verbose:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
        if shuffle:
            if fix_batch:
                self._shuffle_batch_indexes()
            else:
                self._shuffle_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]

        ctx_utts, ctx_lens = [], []
        out_utts, out_lens = [], []
        prev_utts, prev_lens = [], []
        next_utts, next_lens = [], []

        out_bs, out_db = [] , []
        keys = []
        metas = []
        index = []
        action_target = []
        action_name = []
        turn_id = []

        state_table = []
        for row in rows:
            # print(type(row))
            # print(row)
            in_row, out_row, index_row, state_dict = row.context, row.response, row.raw_index, row.state_dict
            
            # source context
            keys.append(row.key)
            index.append(index_row)
            action_target.append(row.action_id)
            turn_id.append(row.turn_id)
            # results = map(int, results)
            batch_ctx = []
            for turn in in_row:
                batch_ctx.append(self.pad_to(self.max_utt_size, turn.utt, do_pad=True))
            ctx_utts.append(batch_ctx)
            ctx_lens.append(len(batch_ctx))

            # target response
            out_utt = [t for idx, t in enumerate(out_row.utt)]
            metas.append('meta')
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))

            prev_utts.append(row.prev_resp.utt)
            prev_lens.append(len(row.prev_resp.utt))

            next_utts.append(row.next_resp.utt)
            next_lens.append(len(row.next_resp.utt))
            state_table.append(state_dict)
        batch_size = len(ctx_lens)

        vec_ctx_lens = np.array(ctx_lens) # (batch_size, ), number of turns
        max_ctx_len = np.max(vec_ctx_lens)
        vec_ctx_utts = np.zeros((batch_size, max_ctx_len, self.max_utt_size), dtype=np.int32)

        vec_out_lens = np.array(out_lens)  # (batch_size, ), number of tokens
        max_out_len = np.max(vec_out_lens)
        vec_out_utts = np.zeros((batch_size, max_out_len), dtype=np.int32)

        vec_prevs = np.zeros((self.batch_size, np.max(prev_lens)), dtype=np.int32)
        vec_nexts = np.zeros((self.batch_size, np.max(next_lens)),dtype=np.int32)
        vec_prev_lens = np.array(prev_lens)
        vec_next_lens = np.array(next_lens)

        for b_id in range(batch_size):
            vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]
            vec_out_utts[b_id, :vec_out_lens[b_id]] = out_utts[b_id]
            vec_prevs[b_id, 0:vec_prev_lens[b_id]] = prev_utts[b_id]
            vec_nexts[b_id, 0:vec_next_lens[b_id]] = next_utts[b_id]


        return Pack(context_lens=vec_ctx_lens, # (batch_size, )
                    contexts=vec_ctx_utts, # (batch_size, max_ctx_len, max_utt_len)
                    output_lens=vec_out_lens, # (batch_size, )
                    outputs=vec_out_utts, # (batch_size, max_out_len)
                    metas=metas,
                    prevs=vec_prevs, 
                    prev_lens=vec_prev_lens,
                    nexts=vec_nexts, 
                    next_lens=vec_next_lens,
                    keys=keys,
                    index=index,
                    action_id=action_target,
                    turn_id=turn_id,
                    state_table=state_table
                    )




class WoZGanDataLoaders(DataLoader):
    def __init__(self, name, config):
        super(WoZGanDataLoaders, self).__init__(name)
        self.config = config
        self.action_num = config.action_num
        data = self._read_file(name)
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data, config.backward_size)        
        self.data_size = len(self.data)
        print("Data size: {}".format(self.data_size))
    
    def _read_file(self, dataset):
        with open(os.path.join('./data/multiwoz', dataset + '.sa.json')) as f:
            data = json.load(f)
        return data
     
    def flatten_dialog(self, data, backward_size):
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
            results.append(Pack(action_id=action_index, 
                                state_onehot=state_onehot,
                                action_id_binary=action_index_binary, 
                                state_convlab=state_convlab))
            indexes.append(len(indexes))
            batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)
        return results, indexes, batch_indexes

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // config.batch_size
        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
        if verbose:
            print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
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

        state_onehot, state_convlab = [], []
        action_id, action_id_binary= [], []

        for row in rows:  
            state_onehot.append(row['state_onehot'])
            state_convlab.append(row['state_convlab'])
            action_id.append(row['action_id'])
            action_id_binary.append(row['action_id_binary'])
            
        # state = np.array(state)
        # action = np.array(action)
        return Pack(action_id=action_id, 
                    state_onehot=state_onehot,
                    state_convlab=state_convlab, 
                    action_id_binary=action_id_binary
                    )




class WoZGanDataLoaders_StateActionEmbed(DataLoader):
    def __init__(self, name, config):
        super(WoZGanDataLoaders_StateActionEmbed, self).__init__(name)
        self.config = config
        self.action_num = config.action_num
        data = self._read_file(name)
        self.data, self.indexes, self.batch_indexes = self.flatten_dialog(data, config.backward_size)        
        self.data_size = len(self.data)
        print("Data size: {}".format(self.data_size))
    
    def _read_file(self, dataset):
        # with open(os.path.join('./data/multiwoz', dataset + '.sa_NoHotel.json')) as f:
        with open(os.path.join('./data/multiwoz', dataset + '.sa_alldomain.json')) as f:           
            data = json.load(f)
        return data
     
    def flatten_dialog(self, data, backward_size):
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
            action_rep_seg = dlg['action_rep_seg']
            results.append(Pack(action_id=action_index, 
                                state_onehot=state_onehot,
                                action_id_binary=action_index_binary, 
                                action_rep_seg=action_rep_seg,
                                state_convlab=state_convlab))
            indexes.append(len(indexes))
            batch_index.append(indexes[-1])
            if len(batch_index) > 0:
                batch_indexes.append(batch_index)
        return results, indexes, batch_indexes

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // config.batch_size
        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])
        if verbose:
            print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
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

        state_onehot, state_convlab = [], []
        action_id, action_id_binary, action_rep = [], [], []

        for row in rows:  
            state_onehot.append(row['state_onehot'])
            state_convlab.append(row['state_convlab'])
            action_id.append(row['action_id'])
            action_id_binary.append(row['action_id_binary'])
            action_rep.append(row['action_rep_seg'])
            
        # state = np.array(state)
        # action = np.array(action)
        return Pack(action_id=action_id, 
                    state_onehot=state_onehot,
                    state_convlab=state_convlab, 
                    action_id_binary=action_id_binary,
                    action_rep_seg=action_rep
                    )


