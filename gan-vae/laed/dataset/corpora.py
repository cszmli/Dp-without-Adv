# -*- coding: utf-8 -*-
# author: Tiancheng Zhao
from __future__ import unicode_literals  # at top of module
from collections import Counter
import numpy as np
import json
from laed.utils import get_tokenize, get_chat_tokenize, missingdict, Pack
import logging
import os
import itertools
from collections import defaultdict
import copy
# sys.path.append('./a3c/src')
import a3c.src.deep_dialog.dialog_config as a3c_config

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'
BOD = "<d>"
EOD = "</d>"
BOT = "<t>"
EOT = "</t>"
ME = "<me>"
OT = "<ot>"
SYS = "<sys>"
USR = "<usr>"
KB = "<kb>"
SEP = "|"
REQ = "<requestable>"
INF = "<informable>"
SEL = '<selection>'
EOTSELF = '<eot>'
WILD = "%s"
SPECIAL_TOKENS_DEAL = [PAD, UNK, USR, SYS, BOD, EOS]
SPECIAL_TOKENS = [PAD, UNK, USR, SYS, BOS, BOD, EOS, EOD, EOTSELF]
STOP_TOKENS = [EOS, SEL]
DECODING_MASKED_TOKENS = [PAD, UNK, USR, SYS, BOD]



class StanfordCorpus(object):
    logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'kvret_train_public.json'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'kvret_dev_public.json'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'kvret_test_public.json'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(path, 'rb') as f:
            data = json.load(f)

        return self._process_dialog(data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        speaker_map = {'assistant': SYS, 'driver': USR}
        for raw_dialog in data:
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta=None)]
            for turn in raw_dialog['dialogue']:
                utt = turn['data']['utterance']
                utt = [BOS, speaker_map[turn['turn']]] + self.tokenize(utt) + [EOS]
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=turn['turn']))

            if hasattr(self.config, 'include_eod') and self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]
        print("<d> index %d" % self.rev_vocab[BOD])

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)


class PTBCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'ptb.train.txt'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'ptb.valid.txt'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'ptb.test.txt'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(path, 'rb') as f:
            lines = f.readlines()

        return self._process_data(lines)

    def _process_data(self, data):
        all_text = []
        all_lens = []
        for line in data:
            tokens = [BOS] + self.tokenize(line)[1:] + [EOS]
            all_lens.append(len(tokens))
            all_text.append(Pack(utt=tokens, speaker=0))
        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        return all_text

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for turn in self.train_corpus:
            all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for line in data:
            id_turn = Pack(utt=self._sent2id(line.utt),
                           speaker=line.speaker,
                           meta=line.get('meta'))
            results.append(id_turn)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)


class DailyDialogCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(os.path.join(self._path, 'train'))
        self.valid_corpus = self._read_file(os.path.join(self._path, 'validation'))
        self.test_corpus = self._read_file(os.path.join(self._path, 'test'))
        self._build_vocab(config.max_vocab_cnt)
        print("Done loading corpus")

    def _read_file(self, path):
        with open(os.path.join(path, 'dialogues.txt'), 'rb') as f:
            txt_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_act.txt'), 'rb') as f:
            da_lines = f.readlines()

        with open(os.path.join(path, 'dialogues_emotion.txt'), 'rb') as f:
            emotion_lines = f.readlines()

        combined_data = [(t, d, e) for t, d, e in zip(txt_lines, da_lines, emotion_lines)]

        return self._process_dialog(combined_data)

    def _process_dialog(self, data):
        new_dialog = []
        bod_utt = [BOS, BOD, EOS]
        eod_utt = [BOS, EOD, EOS]
        all_lens = []
        all_dialog_lens = []
        for raw_dialog, raw_act, raw_emotion in data:
            dialog = [Pack(utt=bod_utt,
                           speaker=0,
                           meta=None)]

            raw_dialog = raw_dialog.decode('ascii', 'ignore').encode()
            raw_dialog = raw_dialog.split('__eou__')[0:-1]
            raw_act = raw_act.split()
            raw_emotion = raw_emotion.split()

            for t_id, turn in enumerate(raw_dialog):
                utt = turn
                utt = [BOS] + self.tokenize(utt.lower()) + [EOS]
                all_lens.append(len(utt))
                dialog.append(Pack(utt=utt, speaker=t_id%2,
                                   meta={'emotion': raw_emotion[t_id], 'act': raw_act[t_id]}))

            if not hasattr(self.config, 'include_eod') or self.config.include_eod:
                dialog.append(Pack(utt=eod_utt, speaker=0))

            all_dialog_lens.append(len(dialog))
            new_dialog.append(dialog)

        print("Max utt len %d, mean utt len %.2f" % (
            np.max(all_lens), float(np.mean(all_lens))))
        print("Max dialog len %d, mean dialog len %.2f" % (
            np.max(all_dialog_lens), float(np.mean(all_dialog_lens))))
        return new_dialog

    def _build_vocab(self, max_vocab_cnt):
        all_words = []
        for dialog in self.train_corpus:
            for turn in dialog:
                all_words.extend(turn.utt)

        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([c for t, c, in vocab_count[max_vocab_cnt:]])
        vocab_count = vocab_count[0:max_vocab_cnt]

        # create vocabulary list sorted by count
        print("Load corpus with train size %d, valid size %d, "
              "test size %d raw vocab size %d vocab size %d at cut_off %d OOV rate %f"
              % (len(self.train_corpus), len(self.valid_corpus),
                 len(self.test_corpus),
                 raw_vocab_size, len(vocab_count), vocab_count[-1][1],
                 float(discard_wc) / len(all_words)))

        self.vocab = [PAD, UNK, SYS, USR] + [t for t, cnt in vocab_count]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
        self.unk_id = self.rev_vocab[UNK]

    def _sent2id(self, sent):
        return [self.rev_vocab.get(t, self.unk_id) for t in sent]

    def _to_id_corpus(self, data):
        results = []
        for dialog in data:
            temp = []
            # convert utterance and feature into numeric numbers
            for turn in dialog:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               meta=turn.get('meta'))
                temp.append(id_turn)
            results.append(temp)
        return results

    def get_corpus(self):
        id_train = self._to_id_corpus(self.train_corpus)
        id_valid = self._to_id_corpus(self.valid_corpus)
        id_test = self._to_id_corpus(self.test_corpus)
        return Pack(train=id_train, valid=id_valid, test=id_test)

class NormMultiWozCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.bs_size = 94
        self.db_size = 30
        self.logger.info(config)
        self.bs_types =['b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'b', 'b', 'b']
        self.domains = ['hotel', 'restaurant', 'train', 'attraction', 'hospital', 'police', 'taxi']
        self.info_types = ['book', 'fail_book', 'fail_info', 'info', 'reqt']
        self.config = config
        self.tokenize = lambda x: x.split()
        self.train_corpus, self.val_corpus, self.test_corpus = self._read_file(self.config)
        self._extract_vocab()
        self._extract_goal_vocab()
        self.logger.info('Loading corpus finished.')
        

    def _read_file(self, config):
        train_data = json.load(open(config.train_path))
        valid_data = json.load(open(config.valid_path))
        test_data = json.load(open(config.test_path))
        
        train_data = self._process_dialogue(train_data)
        valid_data = self._process_dialogue(valid_data)
        test_data = self._process_dialogue(test_data)

        return train_data, valid_data, test_data

    def _process_dialogue(self, data):
        new_dlgs = []
        all_sent_lens = []
        all_dlg_lens = []

        for key, raw_dlg in data.items():
            norm_dlg = [Pack(speaker=USR, utt=[BOS, BOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size)]
            for t_id in range(len(raw_dlg['db'])):
                usr_utt = [BOS] + self.tokenize(raw_dlg['usr'][t_id]) + [EOS]
                sys_utt = [BOS] + self.tokenize(raw_dlg['sys'][t_id]) + [EOS]
                norm_dlg.append(Pack(speaker=USR, utt=usr_utt, db=raw_dlg['db'][t_id], bs=raw_dlg['bs'][t_id]))
                norm_dlg.append(Pack(speaker=SYS, utt=sys_utt, db=raw_dlg['db'][t_id], bs=raw_dlg['bs'][t_id]))
                all_sent_lens.extend([len(usr_utt), len(sys_utt)])
            # To stop dialog
            norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size))
            # if self.config.to_learn == 'usr':
            #     norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size))
            all_dlg_lens.append(len(raw_dlg['db']))
            processed_goal = self._process_goal(raw_dlg['goal'])
            new_dlgs.append(Pack(dlg=norm_dlg, goal=processed_goal, key=key))

        self.logger.info('Max utt len = %d, mean utt len = %.2f' % (
            np.max(all_sent_lens), float(np.mean(all_sent_lens))))
        self.logger.info('Max dlg len = %d, mean dlg len = %.2f' % (
            np.max(all_dlg_lens), float(np.mean(all_dlg_lens))))
        return new_dlgs

    def _extract_vocab(self):
        all_words = []
        for dlg in self.train_corpus:
            for turn in dlg.dlg:
                all_words.extend(turn.utt)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        keep_vocab_size = min(self.config.max_vocab_size, raw_vocab_size)
        oov_rate = np.sum([c for t, c in vocab_count[0:keep_vocab_size]]) / float(len(all_words))

        self.logger.info('cut off at word {} with frequency={},\n'.format(vocab_count[keep_vocab_size - 1][0],
                                                               vocab_count[keep_vocab_size - 1][1]) +
              'OOV rate = {:.2f}%'.format(100.0 - oov_rate * 100))

        vocab_count = vocab_count[0:keep_vocab_size]
        self.vocab = SPECIAL_TOKENS + [t for t, cnt in vocab_count if t not in SPECIAL_TOKENS]
        self.vocab_dict = {t: idx for idx, t in enumerate(self.vocab)}
        self.rev_vocab = self.vocab_dict
        self.unk_id = self.vocab_dict[UNK]
        self.logger.info("Raw vocab size {} in train set and final vocab size {}".format(raw_vocab_size, len(self.vocab)))

    def _process_goal(self, raw_goal):
        res = {}
        for domain in self.domains:
            all_words = []
            d_goal = raw_goal[domain]
            if d_goal:
                for info_type in self.info_types:
                    sv_info = d_goal.get(info_type, dict())
                    if info_type == 'reqt' and isinstance(sv_info, list):
                        all_words.extend([info_type + '|' + item for item in sv_info])
                    elif isinstance(sv_info, dict):
                        all_words.extend([info_type + '|' + k + '|' + str(v) for k, v in sv_info.items()])
                    else:
                        print('Fatal Error!')
                        exit(-1)
            res[domain] = all_words
        return res

    def _extract_goal_vocab(self):
        self.goal_vocab, self.goal_vocab_dict, self.goal_unk_id = {}, {}, {}
        for domain in self.domains:
            all_words = []
            for dlg in self.train_corpus:
                all_words.extend(dlg.goal[domain])
            vocab_count = Counter(all_words).most_common()
            raw_vocab_size = len(vocab_count)
            discard_wc = np.sum([c for t, c in vocab_count])

            self.logger.info('================= domain = {}, \n'.format(domain) +
                  'goal vocab size of train set = %d, \n' % (raw_vocab_size,) +
                  'cut off at word %s with frequency = %d, \n' % (vocab_count[-1][0], vocab_count[-1][1]) +
                  'OOV rate = %.2f' % (1 - float(discard_wc) / len(all_words),))

            self.goal_vocab[domain] = [UNK] + [g for g, cnt in vocab_count]
            self.goal_vocab_dict[domain] = {t: idx for idx, t in enumerate(self.goal_vocab[domain])}
            self.goal_unk_id[domain] = self.goal_vocab_dict[domain][UNK]

    def get_corpus(self):
        id_train = self._to_id_corpus('Train', self.train_corpus)
        id_val = self._to_id_corpus('Valid', self.val_corpus)
        id_test = self._to_id_corpus('Test', self.test_corpus)
        return id_train, id_val, id_test

    def _to_id_corpus(self, name, data):
        results = []
        for dlg in data:
            if len(dlg.dlg) < 1:
                continue
            id_dlg = []
            for turn in dlg.dlg:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               db=turn.db, bs=turn.bs)
                id_dlg.append(id_turn)
            id_goal = self._goal2id(dlg.goal)
            results.append(Pack(dlg=id_dlg, goal=id_goal, key=dlg.key))
        return results

    def _sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]

    def _goal2id(self, goal):
        res = {}
        for domain in self.domains:
            d_bow = [0.0] * len(self.goal_vocab[domain])
            for word in goal[domain]:
                word_id = self.goal_vocab_dict[domain].get(word, self.goal_unk_id[domain])
                d_bow[word_id] += 1.0
            res[domain] = d_bow
        return res

    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

class temp_data(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.temp_corpus = self._read_file(self._path)
    def _read_file(self, path):
        with open(os.path.join(path, 'tem_test.json'), 'rb') as f:
            dialog_lines = json.load(f)
        return dialog_lines


class MovieCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(self._path, 'Train')
        self.valid_corpus = self._read_file(self._path, 'Valid')
        self.test_corpus = self._read_file(self._path, 'Test')
        self._extract_vocab()
        print("Done loading corpus")

    def _read_file(self, path, name):
        with open(os.path.join(path, name+'.movie_all.tsv.json'), 'rb') as f:
            dialog_lines = json.load(f)
        return self._process_dialogue(dialog_lines)

    def _process_dialogue(self, data):
        new_dlgs = []
        all_sent_lens = []
        all_dlg_lens = []

        for session_id, raw_dlg in enumerate(data):
            # norm_dlg = [Pack(speaker=USR, utt=[BOS, BOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size)]
            norm_dlg = []
            if len(raw_dlg)<2:
                continue
            for turn_dialog in raw_dlg:
                utt = [BOS] + self.tokenize(turn_dialog['act_str']) + [EOS]
                # utt = [BOT] + self.tokenize(turn_dialog['act_str']) + [EOT]
                action_id = turn_dialog.get('action_id', 999) 
                if turn_dialog['Message.From']=='user':
                    norm_dlg.append(Pack(speaker=USR, utt=utt, raw_act_id=action_id))
                else:
                    norm_dlg.append(Pack(speaker=SYS, utt=utt, raw_act_id=action_id))
                all_sent_lens.extend([len(utt)])
            # To stop dialog
            # print(norm_dlg)
            if norm_dlg[-1]['speaker']==SYS:
                norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], raw_act_id=999))
                # norm_dlg.append(Pack(speaker=USR, utt=[BOT, EOD, EOT]))

            # if self.config.to_learn == 'usr':
            #     norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size))
            # processed_goal = self._process_goal(raw_dlg['goal'])
            all_dlg_lens.append(len(raw_dlg))
            new_dlgs.append(Pack(dlg=norm_dlg, key=raw_dlg[0]['session']))

        self.logger.info('Max utt len = %d, mean utt len = %.2f' % (
            np.max(all_sent_lens), float(np.mean(all_sent_lens))))
        self.logger.info('Max dlg len = %d, mean dlg len = %.2f' % (
            np.max(all_dlg_lens), float(np.mean(all_dlg_lens))))
        return new_dlgs

    def _extract_vocab(self):
        all_words = []
        for dlg in self.train_corpus:
            for turn in dlg.dlg:
                all_words.extend(turn.utt)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        keep_vocab_size = min(self.config.max_vocab_size, raw_vocab_size)
        oov_rate = np.sum([c for t, c in vocab_count[0:keep_vocab_size]]) / float(len(all_words))

        self.logger.info('cut off at word {} with frequency={},\n'.format(vocab_count[keep_vocab_size - 1][0],
                                                               vocab_count[keep_vocab_size - 1][1]) +
              'OOV rate = {:.2f}%'.format(100.0 - oov_rate * 100))

        vocab_count = vocab_count[0:keep_vocab_size]
        self.vocab = SPECIAL_TOKENS + [t for t, cnt in vocab_count if t not in SPECIAL_TOKENS]
        self.vocab_dict = {t: idx for idx, t in enumerate(self.vocab)}
        self.rev_vocab = self.vocab_dict
        self.unk_id = self.vocab_dict[UNK]
        self.logger.info("Raw vocab size {} in train set and final vocab size {}".format(raw_vocab_size, len(self.vocab)))

    def _process_goal(self, raw_goal):
        res = {}
        for domain in self.domains:
            all_words = []
            d_goal = raw_goal[domain]
            if d_goal:
                for info_type in self.info_types:
                    sv_info = d_goal.get(info_type, dict())
                    if info_type == 'reqt' and isinstance(sv_info, list):
                        all_words.extend([info_type + '|' + item for item in sv_info])
                    elif isinstance(sv_info, dict):
                        all_words.extend([info_type + '|' + k + '|' + str(v) for k, v in sv_info.items()])
                    else:
                        print('Fatal Error!')
                        exit(-1)
            res[domain] = all_words
        return res

    def _extract_goal_vocab(self):
        self.goal_vocab, self.goal_vocab_dict, self.goal_unk_id = {}, {}, {}
        for domain in self.domains:
            all_words = []
            for dlg in self.train_corpus:
                all_words.extend(dlg.goal[domain])
            vocab_count = Counter(all_words).most_common()
            raw_vocab_size = len(vocab_count)
            discard_wc = np.sum([c for t, c in vocab_count])

            self.logger.info('================= domain = {}, \n'.format(domain) +
                  'goal vocab size of train set = %d, \n' % (raw_vocab_size,) +
                  'cut off at word %s with frequency = %d, \n' % (vocab_count[-1][0], vocab_count[-1][1]) +
                  'OOV rate = %.2f' % (1 - float(discard_wc) / len(all_words),))

            self.goal_vocab[domain] = [UNK] + [g for g, cnt in vocab_count]
            self.goal_vocab_dict[domain] = {t: idx for idx, t in enumerate(self.goal_vocab[domain])}
            self.goal_unk_id[domain] = self.goal_vocab_dict[domain][UNK]

    def get_corpus(self):
        id_train = self._to_id_corpus('Train', self.train_corpus)
        id_val = self._to_id_corpus('Valid', self.valid_corpus)
        id_test = self._to_id_corpus('Test', self.test_corpus)
        return id_train, id_val, id_test

    def _to_id_corpus(self, name, data):
        results = []
        for dlg in data:
            if len(dlg.dlg) < 1:
                continue
            id_dlg = []
            for turn in dlg.dlg:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker
                               )
                id_dlg.append(id_turn)
            results.append(Pack(dlg=id_dlg, key=dlg.key))
        return results

    def _sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]


    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

class ArtCorpus(object):
    logger = logging.getLogger()

    def __init__(self, config):
        self.config = config
        self._path = config.data_dir[0]
        self.gamma=0.95
        self.max_utt_len = config.max_utt_len
        self.tokenize = get_chat_tokenize()
        self.train_corpus = self._read_file(self._path, 'train')
        self.valid_corpus = self._read_file(self._path, 'valid')
        self.test_corpus = self._read_file(self._path, 'test')
        self._extract_vocab()
        print("Done loading corpus")
        
    def assign_bucket(self, value):
        if 0.9<=value<1.0:
            return 3
        elif 0.7<=value<0.9:
            return 2
        elif 0.1<=value<0.7:
            return 1
        else:
            return 0

    def init_state_dict(self):
        state_dict = defaultdict(int)
        for slot in a3c_config.slot_set_final:
            state_dict[slot]=0
        return state_dict
    
    def fill_in_state_dict(self, state_dict, utt, role):
        act_list = self.tokenize(utt)
        for word in act_list:
            if word.strip().split('_')[0] in ["confirm_question", "confirm_answer", "thanks", "deny"]:
                word =  word.strip().split('_')[0] + '_'
            entity = role + '_' + word
            if entity in state_dict.keys() and state_dict[entity]==0:
            # if entity in state_dict.keys():            
                state_dict[entity] = 1
            # elif entity not in state_dict.keys():
                # raise ValueError ("no such action: {}".format(entity))
        for key, value in state_dict.items():
            state_dict[key]=state_dict[key] * self.gamma
        return state_dict
    

    def _read_file(self, path, name):
        with open(os.path.join(path, name+'.artificial.json'), 'rb') as f:
            dialog_lines = json.load(f)
        print("Length of {}: {}".format(name, len(dialog_lines)))
        return self._process_dialogue(dialog_lines)

    def _process_dialogue(self, data):
        new_dlgs = []
        all_sent_lens = []
        all_dlg_lens = []

        for session_id, raw_dlg in enumerate(data):
            # norm_dlg = [Pack(speaker=USR, utt=[BOS, BOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size)]
            norm_dlg = []
            state_dict = self.init_state_dict()
            state_dict_feed = [state_dict[k] for k in sorted(state_dict.keys())]
            if len(raw_dlg)<2:
                continue
            for turn_dialog in raw_dlg:
                utt = [BOS] + self.tokenize(turn_dialog['act_str']) + [EOS]
                # utt = self.tokenize(turn_dialog['act_str']) 
                # utt = [BOT] + self.tokenize(turn_dialog['act_str']) + [EOT]
                action_id = turn_dialog.get('action_id', 999) 
                action_turn = turn_dialog.get('turn',0)/2
                if turn_dialog['Message.From']=='user':
                    norm_dlg.append(Pack(speaker=USR, utt=utt, raw_act_id=action_id, turn_num=action_turn, state_dict=state_dict_feed))
                    if self.config.state_type=='table':
                        state_dict = self.fill_in_state_dict(state_dict, turn_dialog['act_str'], 'user') 
                        if self.config.input_type=='sat':
                            state_dict_feed = [self.assign_bucket(state_dict[k]) for k in sorted(state_dict.keys())]
                        else:
                            state_dict_feed = [state_dict[k] for k in sorted(state_dict.keys())]
                else:
                    norm_dlg.append(Pack(speaker=SYS, utt=utt, raw_act_id=action_id, turn_num=action_turn, state_dict=state_dict_feed))
                    if self.config.state_type=='table':
                        state_dict = self.fill_in_state_dict(state_dict, turn_dialog['act_str'], 'agent') 
                        if self.config.input_type=='sat':
                            state_dict_feed = [self.assign_bucket(state_dict[k]) for k in sorted(state_dict.keys())]
                        else:
                            state_dict_feed = [state_dict[k] for k in sorted(state_dict.keys())]
                all_sent_lens.extend([len(utt)])
            # To stop dialog
            # print(norm_dlg)
            if norm_dlg[-1]['speaker']==SYS:
                # norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], raw_act_id=0))
                norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], raw_act_id=0, turn_num=0, state_dict=state_dict_feed))
                # norm_dlg.append(Pack(speaker=USR, utt=[BOT, EOD, EOT]))

            # if self.config.to_learn == 'usr':
            #     norm_dlg.append(Pack(speaker=USR, utt=[BOS, EOD, EOS], bs=[0.0]*self.bs_size, db=[0.0]*self.db_size))
            # processed_goal = self._process_goal(raw_dlg['goal'])
            all_dlg_lens.append(len(raw_dlg))
            new_dlgs.append(Pack(dlg=norm_dlg, key=raw_dlg[0]['session']))

        self.logger.info('Max utt len = %d, mean utt len = %.2f' % (
            np.max(all_sent_lens), float(np.mean(all_sent_lens))))
        self.logger.info('Max dlg len = %d, mean dlg len = %.2f' % (
            np.max(all_dlg_lens), float(np.mean(all_dlg_lens))))
        return new_dlgs

    def _extract_vocab(self):
        all_words = []
        for dlg in self.train_corpus:
            for turn in dlg.dlg:
                all_words.extend(turn.utt)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        keep_vocab_size = min(self.config.max_vocab_size, raw_vocab_size)
        oov_rate = np.sum([c for t, c in vocab_count[0:keep_vocab_size]]) / float(len(all_words))

        self.logger.info('cut off at word {} with frequency={},\n'.format(vocab_count[keep_vocab_size - 1][0],
                                                               vocab_count[keep_vocab_size - 1][1]) +
              'OOV rate = {:.2f}%'.format(100.0 - oov_rate * 100))

        vocab_count = vocab_count[0:keep_vocab_size]
        self.vocab = SPECIAL_TOKENS + [t for t, cnt in vocab_count if t not in SPECIAL_TOKENS]
        self.vocab_dict = {t: idx for idx, t in enumerate(self.vocab)}
        self.rev_vocab = self.vocab_dict
        self.unk_id = self.vocab_dict[UNK]
        self.logger.info("Raw vocab size {} in train set and final vocab size {}".format(raw_vocab_size, len(self.vocab)))

    def _process_goal(self, raw_goal):
        res = {}
        for domain in self.domains:
            all_words = []
            d_goal = raw_goal[domain]
            if d_goal:
                for info_type in self.info_types:
                    sv_info = d_goal.get(info_type, dict())
                    if info_type == 'reqt' and isinstance(sv_info, list):
                        all_words.extend([info_type + '|' + item for item in sv_info])
                    elif isinstance(sv_info, dict):
                        all_words.extend([info_type + '|' + k + '|' + str(v) for k, v in sv_info.items()])
                    else:
                        print('Fatal Error!')
                        exit(-1)
            res[domain] = all_words
        return res

    def _extract_goal_vocab(self):
        self.goal_vocab, self.goal_vocab_dict, self.goal_unk_id = {}, {}, {}
        for domain in self.domains:
            all_words = []
            for dlg in self.train_corpus:
                all_words.extend(dlg.goal[domain])
            vocab_count = Counter(all_words).most_common()
            raw_vocab_size = len(vocab_count)
            discard_wc = np.sum([c for t, c in vocab_count])

            self.logger.info('================= domain = {}, \n'.format(domain) +
                  'goal vocab size of train set = %d, \n' % (raw_vocab_size,) +
                  'cut off at word %s with frequency = %d, \n' % (vocab_count[-1][0], vocab_count[-1][1]) +
                  'OOV rate = %.2f' % (1 - float(discard_wc) / len(all_words),))

            self.goal_vocab[domain] = [UNK] + [g for g, cnt in vocab_count]
            self.goal_vocab_dict[domain] = {t: idx for idx, t in enumerate(self.goal_vocab[domain])}
            self.goal_unk_id[domain] = self.goal_vocab_dict[domain][UNK]

    def get_corpus(self):
        id_train = self._to_id_corpus('Train', self.train_corpus)
        id_val = self._to_id_corpus('Valid', self.valid_corpus)
        id_test = self._to_id_corpus('Test', self.test_corpus)
        return id_train, id_val, id_test

    def _to_id_corpus(self, name, data):
        results = []
        for dlg in data:
            if len(dlg.dlg) < 1:
                continue
            id_dlg = []
            for turn in dlg.dlg:
                id_turn = Pack(utt=self._sent2id(turn.utt),
                               speaker=turn.speaker,
                               raw_act_id=turn.raw_act_id,
                               turn_num=turn.turn_num,
                               state_dict=turn.state_dict
                               )
                id_dlg.append(id_turn)
            results.append(Pack(dlg=id_dlg, key=dlg.key))
        return results

    def _sent2id(self, sent):
        return [self.vocab_dict.get(t, self.unk_id) for t in sent]


    def id2sent(self, id_list):
        return [self.vocab[i] for i in id_list]

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens