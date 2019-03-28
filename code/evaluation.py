import torch
from nltk.translate import bleu_score
from beam import Beam
from config import CUDA, beamsize, n_Best, max_sent_length, hidden_size

def translateBatch(srcBatch, model, src_special, tgt_special):
    srcdata= srcBatch[0]
    src_pad = src_special[0]
    src_eos = src_special[2]

    batchSize = srcBatch[0].size(1)
    beamSize = beamsize
    tt = torch.cuda if CUDA else torch

    mask_src = srcBatch[0].eq(src_pad).permute(1, 0).unsqueeze(0).repeat(beamSize, 1, 1) # 5, batch_size, length
    #  (1) run the encoder on the src
    context, enc_hidden = model.encoder(srcBatch)
    # hi=[-> hi; <- hi]
    h0 = torch.cat((enc_hidden[0], enc_hidden[1]), dim=-1).unsqueeze(0)
    # h1 = torch.cat((enc_hidden[2], enc_hidden[3]), dim=-1).unsqueeze(0)
    # decStates = torch.cat((h0, h1), dim=0)
    decStates = h0

    decStates = decStates.repeat(1, beamSize, 1)  # 1 / 2 * 320 * 512
    context = context.repeat(1, beamSize, 1)

    beam = [Beam(beamSize, tgt_special, CUDA) for k in range(batchSize)]
    batchIdx = list(range(batchSize))  # 1 ~ 64
    remainingSents = batchSize  # 64

    for i in range(max_sent_length):
        # Prepare decoder input.
        input = torch.stack([b.getCurrentState() for b in beam
                             if not b.done]).transpose(0, 1).contiguous().view(1, -1)  # 1 * 320  all is '2'
        input = input.squeeze(0)
        out_prob, decStates, attn = model.decoder(input, decStates, context, mask_src.view(-1, mask_src.size(2)))   # 320 * 20000

        # batch x beam x numWords
        wordLk = out_prob.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()  # 64 * 5 *20000
        attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

        active = []
        father_idx = []
        for b in range(batchSize):
            if beam[b].done:
                continue

            idx = batchIdx[b]
            if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                active += [b]
                father_idx.append(beam[b].prevKs[-1])  # this is very annoying

        if not active:
            break

        # to get the real father index
        real_father_idx = []
        for kk, idx in enumerate(father_idx):
            real_father_idx.append(idx * len(father_idx) + kk)

        # in this section, the sentences that are still active are
        # compacted so that the decoder is not run on completed sentences
        activeIdx = tt.LongTensor([batchIdx[k] for k in active])  # select active batch
        batchIdx = {beam: idx for idx, beam in enumerate(active)}  # beam : 实际的batch id，idx ： active 中的id

        def updateActive(t, rnnSize):
            # select only the remaining active sentences
            view = t.data.view(-1, remainingSents, rnnSize)
            newSize = list(t.size())
            newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents  # reduce batchsize
            return view.index_select(1, activeIdx).view(*newSize)

        # decStates = torch.cat((decStates[0],decStates[1]), dim=-1).unsqueeze(0) # todo:
        decStates = updateActive(decStates, hidden_size)  # 1 * 5*remainingsents * 512*2
        context = updateActive(context, hidden_size)
        mask_src = mask_src.index_select(1, activeIdx)  # 5 * remainingsents * 98

        # set correct state for beam search
        previous_index = torch.stack(real_father_idx).transpose(0, 1).contiguous()  #
        decStates = decStates.view(-1, decStates.size(2)).index_select(0, previous_index.view(-1)).view(
            *decStates.size())
        # decStates = torch.cat((decStates[:,:,:hidden_size],decStates[:,:,hidden_size:]), dim=0) # todo:
        remainingSents = len(active)

    # (4) package everything up
    allHyp, allScores, allattn = [], [], []
    n_best = n_Best

    for b in range(batchSize):
        scores, ks = beam[b].sortBest()
        allScores += [scores[:n_best]]
        valid_attn = srcdata[:, b].ne(src_pad).ne(src_eos).nonzero().squeeze(1)
        hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
        attn = [a.index_select(1, valid_attn) for a in attn] # batch, n_best, len
        allHyp += [hyps]
        allattn += [attn]

    return allHyp, allScores, allattn


def delete_eos(idx, stop):
    idx0 = []
    for i in idx:
        idx0 += [i]
        if i == stop:
            idx0 = idx0[:-1]
            break
    return idx0

def delete_pad(sent):
    pad = '<pad>'
    i = len(sent) - 1
    while sent[i] == pad:
        del sent[i]
        i -=1
    return sent


def evalModel(model, iterator, epoch, rev, src_special, tgt_special, tgt_ref, src_rev):
    tgt_eos = tgt_special[2]

    predict, gold = [], []
    for i, data in enumerate(iterator):
        tgt_data = data.target[0].permute(1, 0)  # batch_size X length
        src_data_wrap = data.source
        ans = data.answer[0]
        src_data = src_data_wrap[0].permute(1, 0)

        if CUDA:
            scr_data = data.source[0].cuda()
            scr_lengths = data.source[1].cuda()
            ans = ans.cuda()
            src_data_wrap = (scr_data, scr_lengths, ans)

        pred, predScore, attn = translateBatch(src_data_wrap, model, src_special, tgt_special)
        # attn: batch, n_best, len_tgt, len_src
        predBatch = []
        for b in range(src_data_wrap[0].size(1)):
            n = 0
            predb = torch.stack(pred[b][n]).cpu()
            predb = delete_eos(predb, tgt_eos)
            att = attn[b][n]
            predb = torch.stack(predb)   # todo: 有empty的情况？ data有问题/逻辑有问题？(几乎很少发生）
            raw = rev.reverse(predb.unsqueeze(1), src_data[b], att, src_rev) # todo: should add post-process?
            predBatch.append(raw[0].split())
        # nltk BLEU evaluator needs tokenized sentences

        # tgt_raw = []
        # for k in range(tgt_data.size(0)):
        #     tgt = rev.reverse(tgt_data[k][1:].unsqueeze(1))
        #     tgt_raw.append(tgt[0].split())
        #
        # gold += [[r] for r in tgt_raw]
        predict += predBatch

    # for i, data in enumerate(val_iter_raw):
    #     tgt_data = data.target[0].permute(1, 0)  # batch_size X length
    #     tgt_raw = []
    #     for k in range(tgt_data.size(0)):
    #         tgt = rev_raw.reverse(tgt_data[k].unsqueeze(1))
    #         tgt_raw.append(tgt[0].split())
    #
    #     gold += [[delete_pad(r)] for r in tgt_raw]

    tgt_writer = open("tgt_output.txt", "w")
    hyp_writer = open("hyp_output.txt", "w")
    for i in tgt_ref:
        tgt_writer.write("__eos__".join([" ".join(i[_id]) for _id in range(len(i))]) + "\n")
    for j in predict:
        hyp_writer.write(" ".join(j) + "\n")


    bleu = bleu_score.corpus_bleu(tgt_ref, predict)
    f = open('bleu' + str(epoch) + '.txt', 'w', encoding='utf-8')
    f.write(str(tgt_ref))
    f.write('             ')
    f.write(str(predict))
    f.close()

    report_metric = bleu

    return report_metric


