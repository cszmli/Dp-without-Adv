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
from collections import defaultdict
from argparse import Namespace
import numpy as np
from laed.utils import cast_type
from torch.autograd import Variable

INT = 0
LONG = 1
FLOAT = 2
logger = logging.getLogger()

def load_context_encoder(agent, pre_sess_path):
    if not os.path.isfile(pre_sess_path):
            print("No context encoder was loaded")
    else:
        encoder_sess = torch.load(pre_sess_path)
        agent.context_encoder.embedding.load_state_dict(encoder_sess['embedding'])
        agent.context_encoder.utt_encoder.load_state_dict(encoder_sess['utt_encoder'])
        agent.context_encoder.ctx_encoder.load_state_dict(encoder_sess['ctx_encoder'])
        agent.p_y.load_state_dict(encoder_sess['p_y'])
        agent.p_fc1.load_state_dict(encoder_sess['p_fc1'])
        # agent.context_encoder.policy_state_norm_layer.load_state_dict(encoder_sess['policy_state_norm_layer'])
        print("Loading context encoder finished!")

def load_context_action_encoder_for_reward_agent(agent, pre_sess_path):
    if not os.path.isfile(pre_sess_path):
            print("No context & action encoder was loaded")
    else:
        encoder_sess = torch.load(pre_sess_path)
        agent.context_encoder.embedding.load_state_dict(encoder_sess['embedding'])
        agent.context_encoder.utt_encoder.load_state_dict(encoder_sess['utt_encoder'])
        agent.context_encoder.ctx_encoder.load_state_dict(encoder_sess['ctx_encoder'])
        print("Loading context encoder finished!")
        # agent.data_exchanger.action_encoder.x_encoder.load_state_dict(encoder_sess['x_encoder'])
        # agent.data_exchanger.action_encoder.x_embedding.load_state_dict(encoder_sess['x_embedding'])
        # agent.data_exchanger.action_encoder.q_y.load_state_dict(encoder_sess['q_y'])
        print("Loading action encoder finished!")

def load_discriminator_for_reward_agent(agent, pre_sess_path):
    if not os.path.isfile(pre_sess_path):
            raise ValueError("No discriminator was loaded")
    else:
        encoder_sess = torch.load(pre_sess_path, map_location='cpu')
        agent.discriminator.load_state_dict(encoder_sess['discriminator'])
        print("Loading discriminator finished!")



def revert_action_dict(id2action):
    action2id = defaultdict(int)
    for i, j in id2action.items():
        action2id[j]=int(i)
    return action2id

def load_gan_agent(agent, pre_sess_path):
    if not os.path.isfile(pre_sess_path):
            raise ValueError("No discriminator was loaded")
    else:
        encoder_sess = torch.load(pre_sess_path)
        agent.discriminator.load_state_dict(encoder_sess['discriminator'])
        print("Loading discriminator finished!")

def save_model(agent, config):
    torch.save(agent.state_dict(), os.path.join(config.session_dir, "model"))
    torch.save({
                # "embedding": agent.context_encoder.embedding.state_dict(),
                # "utt_encoder":agent.context_encoder.utt_encoder.state_dict(),
                # "ctx_encoder": agent.context_encoder.ctx_encoder.state_dict(),
                # "p_y":agent.p_y.state_dict(),
                # "p_fc1":agent.p_fc1.state_dict(),
                "generator":agent.generator.state_dict(),
                "discriminator":agent.discriminator.state_dict()
                # "policy_state_norm_layer":model.policy_state_norm_layer.state_dict()
                },
                os.path.join(config.session_dir, "model_lirl"))
    logger.info("Model Saved.")

def save_model_woz(agent, config):
    torch.save(agent.state_dict(), os.path.join(config.session_dir, "model"))
    torch.save({"generator":agent.generator.state_dict(),
                "discriminator":agent.discriminator.state_dict(),
                "vae":agent.vae.state_dict()
                },
                os.path.join(config.session_dir, "model_lirl"))
    logger.info("Model Saved.")

def save_model_vae(agent, config):
    torch.save({"vae":agent.vae.state_dict()},
                os.path.join(config.session_dir, "model_vae"))
    logger.info("Model Saved.")

def load_model_vae(agent, config):
    if type(config)==str:
        pre_sess_path = os.path.join(config, "model_vae")
    else:
        pre_sess_path = os.path.join(config.session_dir, "model_vae")
    if not os.path.isfile(pre_sess_path):
            raise ValueError("No discriminator was loaded")
    else:
        vae_sess = torch.load(pre_sess_path)
        agent.vae.load_state_dict(vae_sess['vae'])
        print("Loading AutoEncoder finished!")


def load_action2name(config):
    action_file = os.path.join(config.log_dir, config.action2name_path)
    with open(action_file, 'r') as f:
        action_data = json.load(f)
        return action_data

def BCELoss_double(prediction, labels):
    # value = torch.log(prediction) - torch.log(1-prediction)
    value = torch.log(prediction)
    loss = -value * labels
    # loss = -value * (labels) + value * (1-labels)
    return loss.mean()



def cal_accuracy(prediction, labels):
    real_count, real_correct_count=0, 0
    fake_count, fake_correct_count=0, 0
    real_value, fake_value =0, 0
    for pred, lab in zip(prediction, labels):
        if lab==1 and pred>=0.5:
            real_count+=1
            real_correct_count += 1
            real_value += pred
        elif lab==1 and pred<0.5:
            real_count+=1
            real_value += pred
        elif lab==0 and pred>=0.5:
            fake_count+=1
            fake_value += pred
        elif lab==0 and pred<0.5:
            fake_count+=1
            fake_correct_count+=1
            fake_value += pred
    return np.array([real_count, real_correct_count, fake_count, fake_correct_count, 2*real_value/len(labels), 2*fake_value/len(labels)])

def save_data_for_tsne(human_data, machine_data, generator_samples, pred, config):
    mix_rec, human_rec = [], []
    true_label, pred_label = [], [] 
    machine_samples = defaultdict(list)
    for batch in human_data:
        state, action = batch
        for state_line, action_line in zip(state.tolist(), action.tolist()):
            mix_rec.append(state_line+action_line)
            human_rec.append(state_line+action_line)
    for batch in machine_data:
        state, action = batch
        for state_line, action_line in zip(state.cpu().tolist(), action.cpu().tolist()):
            mix_rec.append(state_line+action_line)
    for label_epoch in pred:
        true_label.append([1] * len(human_data) * len(human_data[0][0]) + [0] * len(machine_data) * len(machine_data[0][0]))
        pred_label.append(label_epoch[0] + label_epoch[1])
    
    for epoch_sample in generator_samples:
        epoch_data_list = []
        epoch_id, epoch_data = epoch_sample
        for batch in epoch_data:
            state, action = batch
            for state_line, action_line in zip(state, action):
                epoch_data_list.append(state_line+action_line)
        machine_samples[epoch_id] = epoch_data_list
    
    print("Writing t-SNE files to {}".format(config.session_dir))
    with open(os.path.join(config.session_dir,'tsne.data.json'), 'wb') as f:
        json.dump(mix_rec, f, indent=1)
    with open(os.path.join(config.session_dir,'tsne.human_data.json'), 'wb') as f:
        json.dump(human_rec, f, indent=1)
    with open(os.path.join(config.session_dir,'tsne.samples.json'), 'wb') as f:
        json.dump(machine_samples, f, indent=1)
    with open(os.path.join(config.session_dir,'tsne.true_label.json'), 'wb') as f:
        json.dump(true_label, f, indent=1)
    with open(os.path.join(config.session_dir,'tsne.pred_label.json'), 'wb') as f:
        json.dump(pred_label, f, indent=1)


def print_accuracy(acc_original, num, config):
    if config.gan_type=="gan":
        logger.info("Human-generated: {}, Successful Predictions: {}, Acc: {:.2f}".format(acc_original[0], acc_original[1], acc_original[1]*1.0/acc_original[0]))
        logger.info("Machine-generated: {}, Successful Predictions: {}, Acc: {:.2f}".format(acc_original[2], acc_original[3], acc_original[3]*1.0/acc_original[2]))
        logger.info("Total: {}, Successful Predictions: {}, Acc: {:.2f}".format(acc_original[0]+acc_original[2], acc_original[1]+acc_original[3], (acc_original[1]+acc_original[3])*1.0/(acc_original[0]+acc_original[2])))
        logger.info("D(x) on Human generated: {:.2f}".format(acc_original[4]/num))
        logger.info("D(G(z)) on Machine generated: {:.2f}".format(acc_original[5]/num))
    elif config.gan_type=='wgan':
        logger.info("D(x) on Human generated: {:.3f}".format(acc_original[0]/num))
        logger.info("D(G(z)) on Machine generated: {:.3f}".format(acc_original[1]/num))
