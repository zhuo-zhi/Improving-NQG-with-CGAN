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

train_iter, val_iter, src_special, tgt_special, rev, val_iter_raw, rev_raw, src_rev, src, tgt, ans_special, ans_rev, _= data.train_data()
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

def train_generator_MLE(gen, gen_opt, train_iter, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    best_bleu = 0
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1))
        total_loss = 0
        num_words = 0
        report_loss = 0
        report_num = 0
        for i, data in enumerate(train_iter):
            tgt_data = data.target[0]
            src_data_wrap = data.source
            ans = data.answer[0]

            if CUDA:
                scr_data = data.source[0].to(device)
                scr_lengths = data.source[1].to(device)
                ans = ans.to(device)
                src_data_wrap = (scr_data, scr_lengths, ans)

            tgt_lengths = data.target[1]
            tgt_lengths = torch.LongTensor(tgt_lengths)
            num_words += tgt_lengths.sum().item()

            tgt_data=tgt_data.permute(1,0)   # --> batch x length
            inp, target = helpers.prepare_generator_batch(tgt_data, gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(src_data_wrap, inp, target)   # inp means decoder inp, target means decoder target.
            loss.div(tgt_data.size(1)).backward()
            # loss.backward()
            gen_opt.step()

            report_loss += loss.item()
            report_num += tgt_data.size(1)
            total_loss += loss.item()

            # if i % 20 == -1 % 20:
            #     print(("inter loss = %.4f") % (report_loss / report_num))
            #     report_loss = 0
            #     report_num = 0

        loss_perword = total_loss / num_words
        train_ppl = math.exp(min(loss_perword, 100))
        print('loss  = %.4f' % (total_loss / len(train_iter.dataset)))
        print('ppl  = %.4f' % train_ppl)

        # evaluate blue scores
        # valid data
        # if epoch%5 == -1%5:
        gen.eval()
        # print("Set gen to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
        valid_bleu = evaluation.evalModel(gen, val_iter, epoch, rev, src_special, tgt_special, tgt_ref, src_rev)
        print('Validation bleu-4 = %g' % (valid_bleu * 100))
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            torch.save(gen.state_dict(), 'params.pkl')
            print('save '+str(epoch + 1)+' epoch model')

        gen_opt.updateLearningRate(valid_bleu)
        #train_bleu = evaluation.evalModel(gen, train_iter)
        #print('training bleu = %g' % (train_bleu * 100))
        gen.train()
        # print("Set gen to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))



def train_generator_PG(gen, gen_opt, dis, train_iter, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    global pg_count
    global best_advbleu
    pg_count += 1
    num_sentences = 0
    total_loss = 0
    rollout = Rollout(gen, update_learning_rate)
    for i, data in enumerate(train_iter):
        if i == num_batches:
            break
        src_data_wrap = data.source
        ans = data.answer[0]
        # tgt_data = data.target[0].permute(1, 0)
        passage = src_data_wrap[0].permute(1, 0)

        if CUDA:
            scr_data = data.source[0].to(device)  # lengths x batch_size
            scr_lengths = data.source[1].to(device)
            ans = ans.to(device)
            ans_p = ans.permute(1, 0)
            src_data_wrap = (scr_data, scr_lengths, ans)
            passage = passage.to(device)
            passage = (passage, ans_p)

        num_sentences += scr_data.size(1)
        with torch.no_grad():
            samples, _ = gen.sample(src_data_wrap)        # 64 batch_size works best
            rewards = rollout.get_reward(samples, passage, src_data_wrap, rollout_size, dis, src_rev, rev, train_ref, tgt_pad)

        inp, target = helpers.prepare_generator_batch(samples, gpu=CUDA)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(src_data_wrap, inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()
        total_loss += pg_loss
        rollout.update_params() # TODO: DON'T KNOW WHY

    gen.eval()
    # print("Set gen to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
    valid_bleu = evaluation.evalModel(gen, val_iter, pg_count, rev, src_special, tgt_special, tgt_ref, src_rev)
    print('Validation bleu-4 = %g' % (valid_bleu * 100))
    if valid_bleu > best_advbleu:
        best_advbleu = valid_bleu
        torch.save(gen.state_dict(), 'advparams.pkl')
        print('save model')
    # train_bleu = evaluation.evalModel(gen, train_iter)
    # print('training bleu = %g' % (train_bleu * 100))
    gen.train()

    print("\npg_loss on %d bactches : %.4f" %(i+1, total_loss/num_batches))


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
                target = target.to(device)
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

    d_step = 0
    while(1):
        d_step += 1
        passages = []
        anses = []
        real_samples = []
        fake_samples = []
        real_lengths = []
        fake_lengths = []

        for i, data in enumerate(train_iter):
            if ADV_batches is not None:
                if i+1 == ADV_batches:
                    break

            tgt_data = data.target[0].permute(1, 0)  # batch_size X length
            src_data_wrap = data.source
            ans = data.answer[0]

            if CUDA:
                scr_data = data.source[0].to(device)
                scr_lengths = data.source[1].to(device)
                ans = ans.to(device)
                src_data_wrap = (scr_data, scr_lengths, ans)

            real_sample = tgt_data
            real_length = data.target[1]
            with torch.no_grad():
                fake_sample, fake_length = generator.sample(src_data_wrap)
            fake_sample = fake_sample.cpu()
            fake_length = fake_length.cpu()
            ans = ans.permute(1, 0).cpu()

            # keep lengths as the same in order to pack
            passage = src_data_wrap[0].permute(1, 0)
            pad_len = max_sent_len - passage.size(1)
            m = nn.ConstantPad1d((0, pad_len), src_pad)
            passage = m(passage)
            ans = m(ans)

            # keep lengths as the same in order to pack
            pad_len = max_sent_len - real_sample.size(1)
            m = nn.ConstantPad1d((0, pad_len), tgt_pad)
            real_sample = m(real_sample)

            real_samples.append(real_sample)
            real_lengths.append(real_length)
            fake_samples.append(fake_sample)
            fake_lengths.append(fake_length)
            passages.append(passage)
            anses.append(ans)

        real_samples = torch.cat(real_samples, 0).type(torch.LongTensor)
        real_lengths = torch.cat(real_lengths, 0).type(torch.LongTensor)
        fake_samples = torch.cat(fake_samples, 0).type(torch.LongTensor)
        fake_lengths = torch.cat(fake_lengths, 0).type(torch.LongTensor)
        passages = torch.cat(passages, 0).type(torch.LongTensor)
        anses = torch.cat(anses, 0).type(torch.LongTensor)
        dis_inp, dis_target, dis_len, dis_pa, dis_an = helpers.prepare_discriminator_data(real_samples, real_lengths,
                                                                                   fake_samples, fake_lengths, passages, anses, tgt_special)

        # iterator
        # for i, dis_data in enumerate(dis_iter):
        #     dis_inp = dis_data.question[0]
        #     dis_target = dis_data.target
        #     dis_pa = dis_data.passage[0]
        #     dis_an = dis_data.answer[0]

        # collect discriminator data
        # disc_writer = open("disc.json", "w")
        # question0 = rev.reverse(dis_inp.permute(1,0))
        # answer0 = ans_rev.reverse(dis_an.permute(1, 0))
        # passage0 = src_rev.reverse(dis_pa.permute(1, 0))
        # for i in range(len(dis_inp)):
        #     disc_writer.write("{\"question\": \"" + question0[i][6:] + "\", ")
        #     disc_writer.write("\"answer\": \"" + answer0[i] + "\", ")
        #     disc_writer.write("\"passage\": \"" + passage0[i] + "\", ")
        #     disc_writer.write("\"target\": \"" + str(int(dis_target[i].item())) + "\"}" + "\n")

        # # showcases
        # print(' sample showcase:')
        # show = rev.reverse(dis_inp[:Show_num].permute(1, 0))
        # for i in range(Show_num):
        #     print(show[i])

        for epoch in range(epochs):
            discriminator.train()
            print('\n d-step %d epoch %d : ' % (d_step, epoch + 1), end='')
            total_loss = 0
            total_acc = 0
            true_acc = 0
            num_samples = dis_inp.size(0)

            for i in range(0, num_samples, batch_size):
                inp, target = dis_inp[i: i + batch_size], dis_target[i: i + batch_size]
                # lengths = dis_len[i: i + batch_size]
                pa = dis_pa[i: i + batch_size]
                an = dis_an[i: i + batch_size]
                if CUDA:
                    inp = inp.to(device)
                    target = target.to(device)
                    # lengths = lengths.to(device)
                    an = an.to(device)
                    pa = pa.to(device)
                    pa = (pa, an)

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

                elif d_step+1 == 8 and epoch+1 == 5:
                    flag = 1
                    break

            else:
                if d_step+1 == 4 and epoch+1 == 5:
                    flag = 1
                    break

        if flag == 1:
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

    # GENERATOR MLE TRAINING
    # print('Starting Generator MLE Training...')
    gen_optimizer = Optim('myadam', 1e-3, lr_decay=0.5, start_decay_at=8, max_grad_norm=5)
    gen_optimizer.set_parameters(gen.parameters())
    # train_generator_MLE(gen, gen_optimizer, train_iter, MLE_TRAIN_EPOCHS)

    print('load the best metric model')
    gen.load_state_dict(torch.load('./model/params.pkl'))
    print('evaluating the best model')
    gen.eval()
    # print("Set gen to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
    # valid_bleu = evaluation.evalModel(gen, val_iter, 100, rev, src_special, tgt_special, tgt_ref, src_rev)
    # print('Validation bleu-4 of the best model= %g' % (valid_bleu * 100))

    emb_ans.weight.requires_grad = False
    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR 
    print('\nStarting Discriminator Training...')
    # dis_optimizer = optim.Adam(dis.parameters(), lr=1e-3)
    dis_optimizer = Optim('adam', 1e-3, lr_decay=0.5, max_weight_value=1.0)
    dis_optimizer.set_parameters(dis.parameters())
    # train_discriminator(dis, dis_optimizer, train_iter, gen, pretrain_acc, PRETRAIN_DISC_EPOCHS)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))
    # ADVERSARIAL TRAINING
    pg_count=10000
    best_advbleu = 0

    pg_optimizer = Optim('myadam', 1e-3, max_grad_norm=5)
    pg_optimizer.set_parameters(gen.parameters())
    gen_optimizer.reset_learningrate(1e-3)
    dis_optimizer.reset_learningrate(1e-3)

    rollout = Rollout(gen, update_learning_rate)
    for epoch in range(ADV_TRAIN_EPOCHS):

        for i, data in enumerate(train_iter):
            tgt_data = data.target[0].permute(1, 0)  # batch_size X length
            src_data_wrap = data.source
            ans = data.answer[0]

            if CUDA:
                scr_data = data.source[0].to(device)
                scr_lengths = data.source[1].to(device)
                ans = ans.to(device)
                src_data_wrap = (scr_data, scr_lengths, ans)
                passage = src_data_wrap[0].permute(1, 0)

                tgt_data = tgt_data.to(device)

            if i+1 % 5 != 0:
                gen.eval()
                loss_fn = nn.BCELoss()
                with torch.no_grad():
                    fake_tgt_data, fake_length = gen.sample(src_data_wrap)
                dis_optimizer.zero_grad()
                true_out = dis.batchClassify(tgt_data, (passage, ans)) # hidden = none over here
                true_loss = loss_fn(true_out, torch.zeros([data.batch_size]).type(torch.cuda.FloatTensor))
                fake_out = dis.batchClassify(fake_tgt_data, (passage, ans)) # hidden = none over here
                fake_loss = loss_fn(fake_out, torch.ones([data.batch_size]).type(torch.cuda.FloatTensor))
                loss = true_loss + fake_loss
                loss.backward()
                dis_optimizer.step()
            else:
                gen.train()
                real_sample = tgt_data
                real_length = data.target[1]
                with torch.no_grad():
                    samples, _ = gen.sample(src_data_wrap)  # 64 batch_size works best
                    rewards = rollout.get_reward(samples, (passage, ans ), src_data_wrap, rollout_size, dis, src_rev, rev,
                                                 train_ref, tgt_pad)

                inp, target = helpers.prepare_generator_batch(samples, gpu=CUDA)

                gen_optimizer.zero_grad()
                pg_loss = gen.batchPGLoss(src_data_wrap, inp, target, rewards)
                pg_loss.backward()
                gen_optimizer.step()
                rollout.update_params()  # TODO: DON'T KNOW WHY

        gen.eval()
        # print("Set gen to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
        valid_bleu = evaluation.evalModel(gen, val_iter, pg_count, rev, src_special, tgt_special, tgt_ref, src_rev)
        print('Validation bleu-4 = %g' % (valid_bleu * 100))

        # print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # # TRAIN GENERATOR
        # print('\nAdversarial Training Generator : ', end='')
        # sys.stdout.flush()
        # emb_ans.weight.requires_grad = True
        # gen.train()
        # train_generator_PG(gen, pg_optimizer, dis, train_iter, ADV_pg_batches)
        # # todo: should add teacher forcing traning after PG training?
        # print("teacher forcing training after PG training")
        # train_generator_MLE(gen, gen_optimizer, train_iter, 1)
        #
        # emb_ans.weight.requires_grad = False
        # gen.eval()
        # # TRAIN DISCRIMINATOR
        # print('\nAdversarial Training Discriminator : ')
        # train_discriminator(dis, dis_optimizer, train_iter, gen, ADV_acc, ADV_DISC_EPOCHS, ADV_dis_batches)