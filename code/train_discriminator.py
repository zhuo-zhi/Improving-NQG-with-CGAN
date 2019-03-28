from __future__ import print_function, division
import sys
import math
import torch
import torch.optim as optim
import torch.nn as nn
import generator
import discriminator
import helpers
import evaluation
from data import batch_size
from data import tgt_vocab_size, src_vocab_size
from Optim import Optim
from rollout import Rollout
import reference
import data

CUDA = True
device = torch.device("cuda:0")
# pretrain
MLE_TRAIN_EPOCHS = 10
PRETRAIN_DISC_EPOCHS = 10
# PRETRAN_D_STEPS = 10
pretrain_acc = 0.8
# Show_num = 4
max_sent_len = 100  # for padding

# adv_train
update_learning_rate = 0.8 # policy gradient training
ADV_TRAIN_EPOCHS = 50
ADV_DISC_EPOCHS = 5
ADV_acc = 0.8
ADV_d_step = 1
ADV_pg_batches = int(1000/64) # todo: this setting
rollout_size = 20
ADV_dis_batches = int(10000/64)  # todo: this setting

train_iter, val_iter, src_special, tgt_special, rev, \
val_iter_raw, rev_raw, src_rev, src, tgt, ans_special, ans_rev, disc_train_iter = data.train_data()

tgt_pad = tgt_special[0]
tgt_unk = tgt_special[1]
tgt_eos = tgt_special[2]
tgt_sos = tgt_special[3]
src_pad = src_special[0]
src_unk = src_special[1]
src_eos = src_special[2]
ans_pad = ans_special[0]
ans_unk = ans_special[1]
ans_eos = ans_special[2]

train_ref, tgt_ref = reference.ref(src_rev, rev, rev_raw, train_iter, val_iter_raw)

def train_discriminator(discriminator, dis_opt, train_iter, generator, out_acc, epochs, ADV_batches = None):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """
    def eval(val_iter, discriminator, generator):
        # validation
        discriminator.eval()
        print('validation :', end=' ')
        total_acc = 0
        num_samples = 0
        total_loss = 0
        for i, data in enumerate(val_iter):
            tgt_data = data.target[0].permute(1, 0)  # batch_size X length
            src_data_wrap = data.source
            ans = data.answer[0]

            if CUDA:
                scr_data = data.source[0].to(device)
                scr_lengths = data.source[1].to(device)
                ans = ans.to(device)
                src_data_wrap = (scr_data, scr_lengths, ans)

            real_samples = tgt_data
            real_lengths = data.target[1]
            passage = src_data_wrap[0].permute(1, 0)

            with torch.no_grad():
                fake_samples, fake_lengths = generator.sample(src_data_wrap)
            # prepare prepare_discriminator_data input
            fake_samples = fake_samples.cpu()
            fake_lengths = fake_lengths.cpu()
            ans = ans.permute(1, 0).cpu()

            # shuffle data
            dis_inp, dis_target, dis_len, dis_pa, dis_an = helpers.prepare_discriminator_data(real_samples, real_lengths,
                                                                                     fake_samples, fake_lengths, passage, ans, tgt_special)
            inp, target = dis_inp, dis_target
            lengths, pa = dis_len, dis_pa
            an = dis_an

            if CUDA:
                inp = inp.to(device)
                target = target.to(device).type(torch.float)
                lengths = lengths.to(device)
                pa = pa.to(device)
                an = an.to(device)
                pa = (pa, an)

            # inp = (inp, lengths)
            out = discriminator.batchClassify(inp, pa)
            loss_fn = nn.BCELoss()   # todo: should .cuda??
            loss = loss_fn(out, target)
            total_loss += loss.item()
            num_samples += tgt_data.size(0) * 2
            total_acc += torch.sum((out > 0.5) == (target > 0.5)).item()

        total_acc = total_acc * 1.0 / float(num_samples)
        print('loss = %.4f' % (total_loss / (num_samples)), end=' ')
        print('val_acc = %.4f\n' % (total_acc))
        discriminator.train()
        return total_acc


    for epoch in range(epochs):
        discriminator.train()
        print('\n epoch %d : ' % (epoch + 1), end='')
        total_loss = 0
        total_acc = 0
        true_acc = 0
        num_samples = 0

        for i, dis_data in enumerate(disc_train_iter):
            inp, inp_length = dis_data.question
            target = dis_data.target
            pa, pa_length = dis_data.passage
            ans, ans_length = dis_data.answer
            num_samples += inp.size(1)

            if CUDA:
                pa = pa.transpose(0,1)
                inp = inp.transpose(0,1)
                ans = ans.transpose(0,1)

                inp = inp.to(device)
                target = target.to(device).type(torch.float)
                # lengths = lengths.to(device)
                ans = ans.to(device)
                pa = pa.to(device)
                pa = (pa, ans)


            # inp = (inp, lengths)
            dis_opt.zero_grad()
            out = discriminator.batchClassify(inp, pa) # hidden = none over here
            loss_fn = nn.BCELoss()
            loss = loss_fn(out, target)
            loss.backward()
            dis_opt.step()

            total_loss += loss.item()
            total_acc += torch.sum((out>0.5)==(target>0.5)).item()
            true = (target > 0.5).type(torch.FloatTensor)
            out = out.cpu()
            out_true = out * true
            true_acc += torch.sum(out_true > 0.5).item()

        total_acc = total_acc * 1.0 / float(num_samples)
        true_acc = true_acc * 1.0 / float(num_samples/2)
        print('loss = %.4f, train_acc = %.4f' % (total_loss/(num_samples), total_acc), end=' ')
        print('true_acc = %.4f' % true_acc)
        val_acc = eval(val_iter, discriminator, generator)
        # dis_opt.updateLearningRate(val_acc)


        # todo: when to stop the discriminator MLE training(below is my randomly settings)
        flag = 0
        if ADV_batches is None:
            if val_acc > out_acc:
                flag = 1
                break


# MAIN
if __name__ == '__main__':

    emb_src = nn.Embedding(src_vocab_size + 3, 300, padding_idx=src_pad)
    emb_tgt = nn.Embedding(tgt_vocab_size + 4, 300, padding_idx=tgt_pad)
    emb_ans = nn.Embedding(6, 16, padding_idx=ans_pad)

    emb_src.weight.data.copy_(src.vocab.vectors.to(device))
    emb_tgt.weight.data.copy_(tgt.vocab.vectors.to(device))

    emb_tgt.weight.requires_grad = False
    emb_src.weight.requires_grad = False

    enc = generator.encoder(emb_src, emb_ans, src_special, ans_special)
    dec = generator.decoder(emb_tgt, tgt_special)
    gen = generator.NQGgenerator(enc, dec, gpu=CUDA)
    # dis = discriminator.Discriminator(emb_src, emb_tgt, emb_ans, gpu=CUDA)
    # dis = discriminator.PQANet(emb_src, emb_tgt)
    # dis = discriminator.TransormerNet(emb_src, emb_tgt)
    dis = discriminator.BiLSTM(emb_src, emb_tgt)

    print(dis)

    if CUDA:
        enc = enc.to(device)
        dec = dec.to(device)
        gen = gen.to(device)
        dis = dis.to(device)

    emb_ans.weight.requires_grad = False
    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    # dis_optimizer = optim.Adam(dis.parameters(), lr=1e-3)
    dis_optimizer = Optim('adam', 1e-3, lr_decay=0.5, max_weight_value=1.0)
    dis_optimizer.set_parameters(dis.parameters())
    train_discriminator(dis, dis_optimizer, train_iter, gen, pretrain_acc, PRETRAIN_DISC_EPOCHS)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))
    # ADVERSARIAL TRAINING
    pg_count=10000
    best_advbleu = 0

    pg_optimizer = Optim('myadam', 1e-3, max_grad_norm=5)
    pg_optimizer.set_parameters(gen.parameters())
    gen_optimizer.reset_learningrate(1e-3)
    dis_optimizer.reset_learningrate(1e-3)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        emb_ans.weight.requires_grad = True
        gen.train()
        train_generator_PG(gen, pg_optimizer, dis, train_iter, ADV_pg_batches)
        # todo: should add teacher forcing traning after PG training?
        print("teacher forcing training after PG training")
        train_generator_MLE(gen, gen_optimizer, train_iter, 1)

        emb_ans.weight.requires_grad = False
        gen.eval()
        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, train_iter, gen, ADV_acc, ADV_DISC_EPOCHS, ADV_dis_batches)