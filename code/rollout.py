# -*- coding:utf-8 -*-
import copy
import torch
from nltk.translate import bleu_score
from config import CUDA, lamda, base

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, passage, src_data_wrap, num, discriminator, src_rev, rev, train_ref, tgt_pad):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discriminator model
        """
        x = x[0]
        x_len =x[1]
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        src_data_k = src_data_wrap[0]
        src_data_k = src_rev.reverse(src_data_k)
        for i in range(num):
            for l in range(1, seq_len+1):
                data = x[:, 0:l]
                if data[:, l - 1].eq(tgt_pad).sum() == batch_size:
                    break

                if l < seq_len:
                    samples, lengths = self.own_model.sample(src_data_wrap, 'random', data)
                    samples_wrap = (samples, lengths)
                    pred = discriminator.batchClassify(samples_wrap, passage)
                if l == seq_len:
                    samples = data
                    lengths = x_len
                    samples_wrap = (samples, lengths)
                    pred = discriminator.batchClassify(samples_wrap, passage)

                # bleu-4 reward
                reward_b = []
                s_r = samples.permute(1, 0)
                s_r = rev.reverse(s_r)   # todo: reverse 有unk
                for k in range(samples.size(0)):
                    key = " ".join([idx for idx in src_data_k[k]])
                    score = bleu_score.sentence_bleu(train_ref[key], s_r[k].split())
                    reward_b.append(score)
                reward_b = torch.tensor(reward_b)

                reward = lamda * (pred-base) + (1-lamda) * reward_b.type_as(pred)
                # reward = lamda * (pred-base)
                # reward = reward_b.type_as(pred)

                if i == 0:
                    rewards.append(reward)
                else:
                    rewards[l-1] += reward

        rewards = torch.stack(rewards)
        rewards = rewards / (1.0 * num)  # seq_len * batch_size
        return rewards


    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
