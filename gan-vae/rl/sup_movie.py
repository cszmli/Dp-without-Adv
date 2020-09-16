import time
import os
import json
import torch
import logging
import argparse
from gan.torch_utils import  weights_init
from laed.utils import Pack, set_seed, str2bool, prepare_dirs_loggers, get_time, process_config
import laed.dataset.corpora as corpora
from laed.dataset.data_loaders import MovieDataLoaders
from laed.evaluators import BleuEvaluator
from laed.evaluators_woz import MultiWozEvaluator
from rl.models.models import LIRL
from rl.main import train, validate
import rl.domain as domain
from movie_dialog_utils import task_generate


domain_name = 'object_division'
domain_info = domain.get_domain(domain_name)

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

# Method
method_arg = add_argument_group('Method')
method_arg.add_argument('--gan_type', type=str, default='gan', help="gan_type: gan, wgan")
# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_dir', type=str, nargs='+', default=['data/movie'])
data_arg.add_argument('--train_path', type=str, nargs='+', default='data/movie/Train.movie_all.tsv.json')
data_arg.add_argument('--valid_path', type=str, nargs='+', default='data/movie/Valid.movie_all.tsv.json')
data_arg.add_argument('--test_path', type=str, nargs='+', default='data/movie/Test.movie_all.tsv.json')
data_arg.add_argument('--log_dir', type=str, default='logs_rl')
data_arg.add_argument('--action_id_path', type=str, nargs='+', default='2019-06-04T16-02-movie_vae_laed.py/2019-06-04T16-02-z.pkl.cluster_id.json')
data_arg.add_argument('--action2name_path', type=str, nargs='+', default='2019-06-04T16-02-movie_vae_laed.py/2019-06-04T16-02-z.pkl.id_cluster.json')
data_arg.add_argument('--load_sess', type=str, default="2018-02-04T01-20-45")
data_arg.add_argument('--encoder_sess', type=str, default="2019-06-04T16-02-movie_vae_laed.py")

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--y_size', type=int, default=4)  # number of discrete variables
net_arg.add_argument('--k_size', type=int, default=3)  # number of classes for each variable
net_arg.add_argument('--action_num', type=int, default=55)  
net_arg.add_argument('--gan_ratio', type=int, default=1)  
net_arg.add_argument('--state_noise_dim', type=int, default=256)  
net_arg.add_argument('--action_noise_dim', type=int, default=256)  
net_arg.add_argument('--k', type=int, default=domain_info.input_length())

net_arg.add_argument('--simple_posterior', type=str2bool, default=False)
net_arg.add_argument('--contextual_posterior', type=str2bool, default=True)
net_arg.add_argument('--use_attribute', type=str2bool, default=True)
net_arg.add_argument('--rnn_cell', type=str, default='gru')
net_arg.add_argument('--embed_size', type=int, default=50)

net_arg.add_argument('--utt_type', type=str, default='attn_rnn')
net_arg.add_argument('--utt_cell_size', type=int, default=256)
net_arg.add_argument('--ctx_cell_size', type=int, default=256)
net_arg.add_argument('--dec_cell_size', type=int, default=256)

net_arg.add_argument('--enc_cell_size', type=int, default=256)

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
train_arg.add_argument('--wgan_w_clip', type=float, default=0.01)
train_arg.add_argument('--op', type=str, default='adam')
# train_arg.add_argument('--op', type=str, default='rmsprop')
train_arg.add_argument('--backward_size', type=int, default=5)
train_arg.add_argument('--step_size', type=int, default=1)
# train_arg.add_argument('--grad_clip', type=float, default=3.0)
train_arg.add_argument('--init_w', type=float, default=0.08)
train_arg.add_argument('--init_lr', type=float, default=0.0001)
train_arg.add_argument('--momentum', type=float, default=0.0)
train_arg.add_argument('--lr_hold', type=int, default=1)
train_arg.add_argument('--lr_decay', type=float, default=0.6)
train_arg.add_argument('--l2_lambda', type=float, default=0.00001)
train_arg.add_argument('--dropout', type=float, default=0.3)
train_arg.add_argument('--clip', type=float, default=5.0)
train_arg.add_argument('--improve_threshold', type=float, default=0.998)
train_arg.add_argument('--patient_increase', type=float, default=2.0)
train_arg.add_argument('--early_stop', type=str2bool, default=True)
train_arg.add_argument('--max_epoch', type=int, default=500)
train_arg.add_argument('--max_vocab_size', type=int, default=10000)

train_arg.add_argument('--fix_train_batch', type=str2bool, default=False)

# MISC
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--save_model', type=str2bool, default=True)
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--print_step', type=int, default=500)
misc_arg.add_argument('--fix_batch', type=str2bool, default=False)
misc_arg.add_argument('--train_prior', type=str2bool, default=False)
misc_arg.add_argument('--ckpt_step', type=int, default=1000)
misc_arg.add_argument('--record_step', type=int, default=16)
misc_arg.add_argument('--batch_size', type=int, default=30)
misc_arg.add_argument('--preview_batch_num', type=int, default=1)
misc_arg.add_argument('--gen_type', type=str, default='greedy')
misc_arg.add_argument('--avg_type', type=str, default='word')
misc_arg.add_argument('--beam_size', type=int, default=10)
misc_arg.add_argument('--seed', type=int, default=999)
misc_arg.add_argument('--forward_only', type=str2bool, default=False)


logger = logging.getLogger()

def main(config):
    set_seed(config.seed)
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    stats_path = 'sys_config_log_model'
    if config.forward_only:
        saved_path = os.path.join(stats_path, config.pretrain_folder)
        config = Pack(json.load(open(os.path.join(saved_path, 'config.json'))))
        config['forward_only'] = True
    else:
        saved_path = os.path.join(stats_path, start_time+'-'+os.path.basename(__file__).split('.')[0])
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
    config.saved_path = saved_path

    prepare_dirs_loggers(config)
    logger = logging.getLogger()
    logger.info('[START]\n{}\n{}'.format(start_time, '=' * 30))

    corpus = corpora.MovieCorpus(config)
    train_dial, valid_dial, test_dial = corpus.get_corpus()
    sample_shape = config.batch_size, config.state_noise_dim, config.action_noise_dim
    # evaluator = MultiWozEvaluator("os.path.basename(__file__)")
    evaluator = BleuEvaluator(os.path.basename(__file__))
    # create data loader that feed the deep models
    train_data = MovieDataLoaders("Train", train_dial, config)
    valid_data  = MovieDataLoaders("Valid", valid_dial, config)
    test_data  = MovieDataLoaders("Test", test_dial, config)

    model = LIRL(corpus, config)
    if config.use_gpu:
        model.cuda()

    best_epoch = None
    if not config.forward_only:
        try:
            best_epoch = train(model, train_data, valid_data, test_data, config, evaluator, gen=task_generate)
        except KeyboardInterrupt:
            print('Training stopped by keyboard.')
    if best_epoch is None:
        model_ids = sorted([int(p.replace('-model', '')) for p in os.listdir(saved_path) if 'model' in p and 'rl' not in p])
        best_epoch = model_ids[-1]

    print("$$$ Load {}-model".format(best_epoch))
    config.batch_size = 32
    best_epoch = best_epoch
    model.load_state_dict(torch.load(os.path.join(saved_path, '{}-model'.format(best_epoch))))


    logger.info("Forward Only Evaluation")

    validate(model, valid_data, config)
    validate(model, test_data, config)

    with open(os.path.join(saved_path, '{}_{}_valid_file.txt'.format(start_time, best_epoch)), 'w') as f:
        task_generate(model, valid_data, config, evaluator, num_batch=None, dest_f=f)

    with open(os.path.join(saved_path, '{}_{}_test_file.txt'.format(start_time, best_epoch)), 'w') as f:
        task_generate(model, test_data, config, evaluator, num_batch=None, dest_f=f)

    end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[END]', end_time, '=' * 30)


if __name__ == "__main__":
    config, unparsed = get_config()
    config = process_config(config)
    main(config)
