import torch
from torch.autograd import Variable
from config import device

def prepare_generator_batch(target_batch, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    target = target_batch[:,1:]
    inp= target_batch[:,:-1]
    
    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.to(device)
        target = target.to(device)

    return inp, target


def prepare_discriminator_data(pos_samples, pos_lengths, neg_samples, neg_lengths, passages, ans, src_lens, tgt_special):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """
    tgt_pad = tgt_special[0]

    pos_seq_len = pos_samples.size(1)
    neg_seq_len = neg_samples.size(1)
    max_seq_len = max(pos_seq_len, neg_seq_len)

    num_pos = len(pos_samples)
    num_neg = len(neg_samples)
    num_samples = num_pos + num_neg

    inp = torch.ones(num_samples, max_seq_len).type(torch.LongTensor) * tgt_pad
    inp[:num_pos, :pos_seq_len] = pos_samples
    inp[num_pos:, :neg_seq_len] = neg_samples

    passages = passages.repeat(2, 1)
    ans = ans.repeat(2, 1)
    src_lens = src_lens.repeat(2)

    lengths = torch.cat((pos_lengths, neg_lengths), 0)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[:pos_samples.size()[0]] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]
    lengths = lengths[perm]
    passages = passages[perm]
    ans = ans[perm]
    src_lens = src_lens[perm]
    inp = Variable(inp)
    target = Variable(target)


    return inp, target, lengths, passages, ans, src_lens


