# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from laed.models.model_bases import summary
import torch
from laed.dataset.corpora import PAD, EOS, EOT
from laed.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from laed.utils import get_dekenize
import os
from collections import defaultdict
import logging
from torch_utils import weights_init, adjust_learning_rate, one_hot_embedding, HistoryData
from utils import print_accuracy, save_model, save_model_woz, save_model_vae, load_model_vae
from gan_validate import disc_validate, vae_validate, gen_validate, policy_validate_for_human, LossManager, disc_validate_for_tsne, disc_train_history, disc_validate_for_tsne_single_input, disc_validate_for_tsne_state_action_embed
logger = logging.getLogger()

def train_disc_with_history(agent, history_pool, batch, sample_shape, disc_optimizer, batch_cnt):
    for _ in range(1):
        disc_optimizer.zero_grad()
        if len(history_pool.experience_pool)<1:
            break
        fake_s_a = history_pool.next()
        disc_loss, _ = agent.disc_train(sample_shape, batch, fake_s_a)
        # disc_loss, train_acc = agent.disc_train(sample_shape, batch)
        agent.discriminator.backward(batch_cnt, disc_loss)
        disc_optimizer.step()

def gan_train(agent, machine_data, train_feed, valid_feed, test_feed, config, evaluator, pred_list=[], gen_sampled_list=[]):
    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    batch_cnt = 0
    vae_flag = False
    disc_optimizer = agent.discriminator.get_optimizer(config)
    generator_optimizer = agent.generator.get_optimizer(config)
    if config.domain=='multiwoz' and vae_flag:
        generator_vae_optimizer = agent.gan_vae_optimizer(config)
    
    optimizer_com = torch.optim.Adam(agent.parameters(), lr=0)
    history_pool = HistoryData(10000)
    done_epoch = 0
    train_loss = LossManager()
    agent.train()
    if config.domain=='multiwoz' and vae_flag:
        agent.vae.eval()
    
    logger.info("**** Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))
    disc_on_random_data, epoch_valid = [1.0, 1.0], 0
    largest_diff = -1.0
    # best_valid_loss = 100
    while True:
        train_feed.epoch_init(config, verbose=done_epoch==0, shuffle=True)
        batch_count_inside=-1
        # adjust_learning_rate(disc_optimizer, done_epoch, config)
        # adjust_learning_rate(generator_optimizer, done_epoch, config)

        # logger.info("Train {}/{}".format(done_epoch, config.max_epoch))
        while True:
            if config.domain=='multiwoz' and vae_flag:
                agent.vae.eval()
            batch_count_inside+=1
            batch = train_feed.next_batch()
            sample_shape = config.batch_size, config.state_noise_dim, config.action_noise_dim
            if batch is None:
                agent.discriminator.decay_noise()
                break
            
            # fix the context encoder output
            # agent.context_encoder.eval()
                ''''''''''''''' Training discriminator '''''''''''''''
            for _ in range(config.gan_ratio):
                if config.gan_type=='wgan':
                    for p in agent.discriminator.parameters():
                        p.data.clamp_(-0.03, 0.03)
                disc_optimizer.zero_grad()
                optimizer_com.zero_grad()
                disc_loss, train_acc = agent.disc_train(sample_shape, batch)                
                agent.discriminator.backward(batch_cnt, disc_loss)
                disc_optimizer.step()
                train_disc_with_history(agent, history_pool, batch, sample_shape, disc_optimizer, batch_cnt)
                
            ''''''''''''''' Training generator '''''''''''''''
            # '''
            generator_optimizer.zero_grad()
            if config.domain=='multiwoz' and vae_flag:
                generator_vae_optimizer.zero_grad()
            gen_loss, fake_s_a = agent.gen_train(sample_shape)
            agent.generator.backward(batch_cnt, gen_loss)
            # generator_vae_optimizer.step()            
            generator_optimizer.step()
            history_pool.add(fake_s_a)
            # '''

        
            batch_cnt += 1
            train_loss.add_loss(disc_loss)

            if batch_count_inside == 0 and done_epoch % 1==0:
                logger.info("\n**** Epcoch {}/{} Done ****".format(done_epoch, config.max_epoch))
                logger.info("\n=== Evaluating Model ===")
                logger.info(train_loss.pprint("Train"))
                logger.info("Average disc value for human and machine on training set: {:.3f}, {:.3f}".format(train_acc[-2], train_acc[-1]))
                
                # validation
                agent.eval()
                logger.info("====Validate Discriminator====")
                valid_loss_disc = disc_validate(agent,valid_feed, config, sample_shape, batch_cnt)
                logger.info("====Validate Generator====")
                valid_loss, gen_samples = gen_validate(agent,valid_feed, config, sample_shape, done_epoch, batch_cnt)
                if len(gen_samples)>0:
                    gen_sampled_list.append([done_epoch, gen_samples])
                # logger.info("====Validate Discriminator for t-SNE====")
                # you can skip this step because it is just an additional way to validate the disc.
                if agent.vae.vae_in_size==392:
                    pred, disc_value = disc_validate_for_tsne_single_input(agent, machine_data, valid_feed, config, sample_shape)  
                elif agent.vae.vae_in_size==492:
                    pred, disc_value = disc_validate_for_tsne_state_action_embed(agent, machine_data, valid_feed, config, sample_shape) 
             
                pred_list.append(pred)


                if config.save_model:         
                    save_model_woz(agent, config) 
                        
                disc_on_random_data = disc_value
                epoch_valid = done_epoch
                best_valid_loss = valid_loss
                largest_diff = disc_value[0] - disc_value[1]
                        
                config.early_stop = False
                if done_epoch >= config.max_epoch \
                        or config.early_stop and patience <= done_epoch:
                    if done_epoch < config.max_epoch:
                        logger.info("!!Early stop due to run out of patience!!")

                    logger.info("Best validation loss %f" % best_valid_loss)
                    logger.info("Best validation Epoch on Machine data: {}".format(epoch_valid))
                    logger.info("Best validation Loss: {}".format(best_valid_loss))                    
                    logger.info("Best validation value on Machine data: {}, {}".format(disc_on_random_data[0], disc_on_random_data[1]))
                    
                    return

                # exit eval model
                agent.train()
                train_loss.clear()
                
        done_epoch += 1   
        

def vae_train(agent, train_feed, valid_feed, test_feed, config):
    patience = 10  # wait for at least 10 epoch before stop
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    batch_cnt = 0
    vae_optimizer = agent.vae.get_optimizer(config)
    train_loss = LossManager()
    agent.vae.train()
    done_epoch = 0
    logger.info("**** VAE Training Begins ****")
    logger.info("**** Epoch 0/{} ****".format(config.max_epoch))
    disc_on_random_data, epoch_valid = [1.0, 1.0], 0
    largest_diff = -1.0
    while True:
        train_feed.epoch_init(config, verbose=done_epoch==0, shuffle=True)
        batch_count_inside=-1
        # adjust_learning_rate(disc_optimizer, done_epoch, config)
        # adjust_learning_rate(generator_optimizer, done_epoch, config)

        # logger.info("Train {}/{}".format(done_epoch, config.max_epoch))
        while True:
            batch_count_inside+=1
            batch = train_feed.next_batch()
            if batch is None:
                break
            vae_optimizer.zero_grad()
            vae_loss= agent.vae_train(batch)
            agent.vae.backward(batch_cnt, vae_loss)
            vae_optimizer.step()

            batch_cnt += 1
            train_loss.add_loss(vae_loss)
            # train_loss.add_loss(disc_loss_self)

            if batch_count_inside==0:
                logger.info("\n@@@@@@@@@@ AutoEncoder Epoch: {} % {} @@@@@@@@@@@@".format(done_epoch, config.max_epoch))
                logger.info(train_loss.pprint("Train"))
                valid_loss = vae_validate(agent,valid_feed, config, batch_cnt)
                # update early stopping stats
                if valid_loss < best_valid_loss:
                    if valid_loss <= valid_loss_threshold * config.improve_threshold:
                        patience = max(patience,
                                       done_epoch * config.patient_increase)
                        valid_loss_threshold = valid_loss
                        logger.info("Update patience to {}".format(patience))


                    if config.save_model:                    
                        save_model_vae(agent, config)
                    
                    best_valid_loss = valid_loss
                        

                if done_epoch >= config.max_epoch \
                        or config.early_stop and patience <= done_epoch:
                    if done_epoch < config.max_epoch:
                        logger.info("!!Early stop due to run out of patience!!")

                    logger.info("Best validation Loss: {}".format(best_valid_loss))                    
                    return True

                # exit eval model
                agent.train()
                train_loss.clear()
                
        done_epoch += 1   
