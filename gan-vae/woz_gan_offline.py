# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging
import os
import json
import torch

from laed import evaluators, utt_utils, dialog_utils
from gan import gan_main as engine
from laed.dataset import corpora
from laed.dataset import data_loaders
from laed.models import sent_models
from laed.models import dialog_models
from gan.gan_agent import GanRnnAgent
from gan.utils import load_context_encoder
from laed.utils import str2bool, prepare_dirs_loggers, get_time, process_config
from laed.dataset.data_loaders import BeliefDbDataLoaders
from laed.evaluators_woz import MultiWozEvaluator

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, nargs='+', default=['data/norm-multi-woz'])
data_arg.add_argument('--train_path', type=str, nargs='+', default='data/norm-multi-woz/train_dials.json')
data_arg.add_argument('--valid_path', type=str, nargs='+', default='data/norm-multi-woz/val_dials.json')
data_arg.add_argument('--test_path', type=str, nargs='+', default='data/norm-multi-woz/test_dials.json')
data_arg.add_argument('--log_dir', type=str, default='logs')
data_arg.add_argument('--action_id_path', type=str, nargs='+', default='2019-06-01T22-45-37-woz_vae_laed.py/2019-06-01T22-46-03-z.pkl.cluster_id.json')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--y_size', type=int, default=6)  # number of discrete variables
net_arg.add_argument('--k', type=int, default=3)  # number of classes for each variable
net_arg.add_argument('--action_num', type=int, default=573)  
net_arg.add_argument('--gan_ratio', type=int, default=5)  
net_arg.add_argument('--state_noise_dim', type=int, default=512)  
net_arg.add_argument('--action_noise_dim', type=int, default=512)  

net_arg.add_argument('--use_attribute', type=str2bool, default=True)
net_arg.add_argument('--rnn_cell', type=str, default='gru')
net_arg.add_argument('--embed_size', type=int, default=200)

net_arg.add_argument('--utt_type', type=str, default='attn_rnn')
net_arg.add_argument('--utt_cell_size', type=int, default=256)
net_arg.add_argument('--ctx_cell_size', type=int, default=512)
net_arg.add_argument('--dec_cell_size', type=int, default=512)

net_arg.add_argument('--enc_cell_size', type=int, default=512)

net_arg.add_argument('--bi_ctx_cell', type=str2bool, default=False)
net_arg.add_argument('--bi_enc_cell', type=str2bool, default=False)
net_arg.add_argument('--max_utt_len', type=int, default=40)
net_arg.add_argument('--max_dec_len', type=int, default=40)
net_arg.add_argument('--max_vocab_cnt', type=int, default=10000)
net_arg.add_argument('--num_layer', type=int, default=1)

net_arg.add_argument('--use_attn', type=str2bool, default=False)
net_arg.add_argument('--attn_type', type=str, default='cat')
net_arg.add_argument('--use_mutual', type=str2bool, default=True)
net_arg.add_argument('--use_reg_kl', type=str2bool, default=True)
net_arg.add_argument('--greedy_q', type=str2bool, default=True)


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--op', type=str, default='adam')
train_arg.add_argument('--backward_size', type=int, default=5)
train_arg.add_argument('--step_size', type=int, default=1)
train_arg.add_argument('--grad_clip', type=float, default=3.0)
train_arg.add_argument('--init_w', type=float, default=0.08)
train_arg.add_argument('--init_lr', type=float, default=0.001)
train_arg.add_argument('--momentum', type=float, default=0.0)
train_arg.add_argument('--lr_hold', type=int, default=1)
train_arg.add_argument('--lr_decay', type=float, default=0.6)
train_arg.add_argument('--dropout', type=float, default=0.3)
train_arg.add_argument('--improve_threshold', type=float, default=0.998)
train_arg.add_argument('--patient_increase', type=float, default=2.0)
train_arg.add_argument('--early_stop', type=str2bool, default=False)
train_arg.add_argument('--max_epoch', type=int, default=1000)
train_arg.add_argument('--max_vocab_size', type=int, default=1000)

# MISC
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--print_step', type=int, default=500)
misc_arg.add_argument('--fix_batch', type=str2bool, default=False)
misc_arg.add_argument('--train_prior', type=str2bool, default=False)
misc_arg.add_argument('--ckpt_step', type=int, default=2000)
misc_arg.add_argument('--batch_size', type=int, default=30)
misc_arg.add_argument('--preview_batch_num', type=int, default=1)
misc_arg.add_argument('--gen_type', type=str, default='greedy')
misc_arg.add_argument('--avg_type', type=str, default='word')
misc_arg.add_argument('--beam_size', type=int, default=10)
misc_arg.add_argument('--forward_only', type=str2bool, default=False)
data_arg.add_argument('--load_sess', type=str, default="2018-02-04T01-20-45")
data_arg.add_argument('--encoder_sess', type=str, default="2019-06-01T22-45-37-woz_vae_laed.py")


logger = logging.getLogger()


def main(config):
    prepare_dirs_loggers(config, os.path.basename(__file__))

    corpus_client = corpora.NormMultiWozCorpus(config)

    dial_corpus = corpus_client.get_corpus()
    train_dial, valid_dial, test_dial = dial_corpus
    sample_shape = config.batch_size, config.state_noise_dim, config.action_noise_dim
    # evaluator = evaluators.BleuEvaluator("os.path.basename(__file__)")
    evaluator = MultiWozEvaluator('SysWoz')
    # create data loader that feed the deep models
    train_feed = data_loaders.BeliefDbDataLoaders("Train", train_dial, config)
    valid_feed = data_loaders.BeliefDbDataLoaders("Valid", valid_dial, config)
    test_feed = data_loaders.BeliefDbDataLoaders("Test", test_dial, config)
    model = GanRnnAgent(corpus_client, config)
    load_context_encoder(model, os.path.join(config.log_dir, config.encoder_sess, "model_lirl"))

    if config.forward_only:
        test_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.log_dir, config.load_sess,
                                 "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.log_dir, config.load_sess, "model")
    else:
        test_file = os.path.join(config.session_dir,
                                 "{}-test-{}.txt".format(get_time(), config.gen_type))
        dump_file = os.path.join(config.session_dir, "{}-z.pkl".format(get_time()))
        model_file = os.path.join(config.session_dir, "model")
    
    

    if config.use_gpu:
        model.cuda()
    
    print("Evaluate initial model on Validate set")
    engine.disc_validate(model, valid_feed, config, sample_shape)
    print("Start training")

    if config.forward_only is False:
        try:
            engine.gan_train(model, train_feed, valid_feed, test_feed, config, evaluator)
        except KeyboardInterrupt:
            print("Training stopped by keyboard.")
    print("Trainig Done! Start Testing")
    model.load_state_dict(torch.load(model_file))
    engine.disc_validate(model, valid_feed, config, sample_shape)
    engine.disc_validate(model, test_feed, config, sample_shape)

    # dialog_utils.generate_with_adv(model, test_feed, config, None, num_batch=None)
    # selected_clusters, index_cluster_id_train = utt_utils.latent_cluster(model, train_feed, config, num_batch=None)
    # _, index_cluster_id_test = utt_utils.latent_cluster(model, test_feed, config, num_batch=None)
    # _, index_cluster_id_valid = utt_utils.latent_cluster(model, valid_feed, config, num_batch=None)
    # selected_outs = dialog_utils.selective_generate(model, test_feed, config, selected_clusters)
    # print(len(selected_outs))
    '''
    with open(os.path.join(dump_file+'.json'), 'wb') as f:
        json.dump(selected_clusters, f, indent=2)

    with open(os.path.join(dump_file+'.cluster_id.json.Train'), 'wb') as f:
        json.dump(index_cluster_id_train, f, indent=2)
    with open(os.path.join(dump_file+'.cluster_id.json.Test'), 'wb') as f:
        json.dump(index_cluster_id_test, f, indent=2)
    with open(os.path.join(dump_file+'.cluster_id.json.Valid'), 'wb') as f:
        json.dump(index_cluster_id_valid, f, indent=2)

    with open(os.path.join(dump_file+'.out.json'), 'wb') as f:
        json.dump(selected_outs, f, indent=2)

    with open(os.path.join(dump_file), "wb") as f:
        print("Dumping test to {}".format(dump_file))
        dialog_utils.dump_latent(model, test_feed, config, f, num_batch=None)

    with open(os.path.join(test_file), "wb") as f:
        print("Saving test to {}".format(test_file))
        dialog_utils.gen_with_cond(model, test_feed, config, num_batch=None,
                                   dest_f=f)

    with open(os.path.join(test_file+'.txt'), "wb") as f:
        print("Saving test to {}".format(test_file))
        dialog_utils.generate(model, test_feed, config, evaluator, num_batch=None,
                                   dest_f=f)
    '''


if __name__ == "__main__":
    config, unparsed = get_config()
    config = process_config(config)
    main(config)
