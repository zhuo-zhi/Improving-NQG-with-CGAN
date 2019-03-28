import torch

CUDA=True
device = torch.device("cuda:0")
# generator
max_sent_len = 100 # sample
hidden_size = 512

# beam_search
beamsize = 5
n_Best = 1
max_sent_length = 35 # todo: maybe small


# rollout_reward_setting
lamda = 0.8
base = 0.5

# data_setting
tgt_vocab_size = 28000
src_vocab_size = 45000
batch_size = 64
data_name = 86636

