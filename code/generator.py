import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from data import tgt_vocab_size, src_vocab_size
import torch.nn.functional as F
from config import max_sent_len, hidden_size

class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask=None):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        if x_mask is not None:
            xWy.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(xWy, dim=-1)
        return alpha


class encoder(nn.Module):
    def __init__(self, emb_src, emb_ans, src_special, ans_special):
        super(encoder, self).__init__()
        self.layers = 1
        self.num_directions = 2 
        self.hidden_size = int(hidden_size / self.num_directions)
        self.src_vocab_size = src_vocab_size + 3  # add pad, unk, eos
        self.emb_dim=300
        self.ans_dim=16
        self.dropout=0.3
        self.src_pad = src_special[0]
        self.src_unk = src_special[1]
        self.src_eos = src_special[2]
        self.ans_pad = ans_special[0]
        self.ans_unk = ans_special[1]
        self.ans_eos = ans_special[2]
        self.embedding = emb_src
        self.emb_ans = emb_ans
        self.rnn = nn.GRU(self.emb_dim + self.ans_dim, self.hidden_size,
                          num_layers=self.layers,
                          dropout=self.dropout,
                          bidirectional=True)

    def forward(self, input, hidden=None):
        """
        input: (source, lengths, ans)
        """
        lengths = input[1].data.tolist()  # lengths data is wrapped inside a Variable
        input_emb = self.embedding(input[0])   # input_emb: length X batch_size X vocab_size
        ans_emb = self.emb_ans(input[2])
        input_emb = torch.cat((input_emb, ans_emb), dim=-1)

        emb = pack(input_emb, lengths)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]   # outputs: length X batch_size X hidden_size
        return outputs, hidden_t

class decoder(nn.Module):
    def __init__(self, emb_tgt, tgt_special):
        super(decoder, self).__init__() 
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size + 4  # add pad, unk, eos, sos
        self.emb_dim = 300
        self.dropout = 0.3
        self.tgt_pad = tgt_special[0]
        self.tgt_unk = tgt_special[1]
        self.tgt_eos = tgt_special[2]
        self.tgt_sos = tgt_special[3]

        self.embedding = emb_tgt
        self.gru = nn.GRU(self.emb_dim, self.hidden_size, dropout=self.dropout, num_layers=self.n_layers)
        # self.gru = StackedGRU(self.n_layers, self.emb_dim, self.hidden_size, self.dropout)
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.attn = BilinearSeqAttn(self.hidden_size, self.hidden_size)

    def _forward(self, input, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                    # batch_size
        input = self.embedding(input)  # batch, emb_dim
        input = input.view(1, -1, self.emb_dim)  # 1, batch, hidden
        output, hidden = self.gru(input, hidden) # 1, batch, hidden(out)
        output = self.out(output.view(-1, self.hidden_size)) # batch_size x vocab_size
        output = self.softmax(output)
        return output, hidden

    def forward(self, input, hidden, source_hiddens, mask_src):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                    # batch_size
        input = self.embedding(input)  # batch, emb_dim
        input = input.view(1, -1, self.emb_dim)  # 1 / 2, batch, hidden
        output, hidden = self.gru(input, hidden) # 1 / 2, batch, hidden(out)

        source_mask = mask_src
        source_hiddens = source_hiddens.transpose(0, 1).contiguous()
        scores = self.attn(source_hiddens, output.squeeze(0), source_mask) # batch * len
        context = scores.unsqueeze(1).bmm(source_hiddens).squeeze(1)

        output = self.out(F.tanh(self.linear(torch.cat((context, output.squeeze(0)), 1)))) # batch_size x vocab_size
        output = self.softmax(output)
        return output, hidden, scores


class NQGgenerator(nn.Module):
    def __init__(self, encoder, decoder, gpu=False):
        super(NQGgenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.tgt_sos = decoder.tgt_sos
        self.tgt_pad = decoder.tgt_pad
        self.tgt_eos = decoder.tgt_eos
        self.src_pad = encoder.src_pad
        self.src_eos = encoder.src_eos

        self.max_seq_len = max_sent_len
        self.tgt_vocab_size = tgt_vocab_size + 4   # decoder side
        self.gpu = gpu
    
    def forward(self, letter, dec_hidden, enc_hiddens, mask_src):    # dec_hidden is from encoder
        out, hidden, _ = self.decoder(letter, dec_hidden, enc_hiddens, mask_src)
        return out, hidden
        # out, hidden, att = self.decoder(letter, dec_hidden, enc_hiddens, mask_src)
        # return out, hidden, att


    def _sample(self, batch_size, seq_len, x=None):
        res = []
        flag = False # whether sample from zero
        if x is None:
            flag = True
        if flag:
            x = Variable(torch.zeros((batch_size, 1)).long())
        if self.use_cuda:
            x = x.cuda()
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            for i in range(seq_len):
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                output, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = output.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                output, h, c = self.step(x, h, c)
                x = output.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output

    def sample(self, src_input, way, x=None): # todo: 每次都要经过encoder太浪费时间了。
        """
        x : (batch_size, seq_len) input data
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)

        """
        num_samples = len(src_input[1])
        samples = torch.ones(num_samples, self.max_seq_len).type(torch.LongTensor) * self.tgt_pad
        # add sos
        if x is None:
            samples[:, 0] = torch.ones(num_samples) * self.tgt_sos
            lengths = torch.ones(num_samples).type(torch.LongTensor)
        else:
            given_len = x.size(1)
            samples[:, :given_len] = x
            lengths = torch.ones(num_samples).type(torch.LongTensor)
            for m in range(num_samples):
                for p in range(given_len):
                    if samples[m][p] == self.tgt_eos:
                        lengths[m] = p+1
                        break


        mask_src = src_input[0].eq(self.src_pad).permute(1, 0)
        context, enc_hidden = self.encoder(src_input)
        # hi=[-> hi; <- hi]
        h = torch.cat((enc_hidden[0], enc_hidden[1]), dim=-1).unsqueeze(0)    # h: 1 X batch_size X hidden_size
        # h1 = torch.cat((enc_hidden[2], enc_hidden[3]), dim=-1).unsqueeze(0)
        # h = torch.cat((h0, h1), dim=0)
        # inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

        if self.gpu:
            samples = samples.cuda()
            # inp = inp.cuda()
            h = h.cuda()
            lengths = lengths.cuda()

        # for j in range(num_samples):
        #     h_j = h[:, j, :].unsqueeze(1)
        #     inp_j = inp[j]
        #     context_j = context[:, j, :].unsqueeze(1)
        #     mask_src_j = mask_src[j].unsqueeze(0)
        #     lengths[j] = self.max_seq_len
        #
        #     for i in range(1, self.max_seq_len):
        #         out_j, h_j = self.forward(inp_j, h_j, context_j, mask_src_j)           # out: num_samples x vocab_size
        #         out_j = torch.multinomial(torch.exp(out_j), 1)  # sampling from j row
        #         samples[j, i] = out_j.squeeze()
        #         if samples[j, i] == tgt_eos:
        #             lengths[j] = i + 1
        #             break
        #         else:
        #             inp_j = out_j.view(-1)

        if x is None:
            for i in range(1, self.max_seq_len):
                out, h = self.forward(samples[:,i-1], h, context, mask_src)
                if way == 'random':
                # random sampling
                    out_indexes = torch.multinomial(torch.exp(out), 1)
                if way == 'greedy':
                # greedy sampling
                    out_indexes = torch.exp(out).max(1)[1].unsqueeze(1)

                if samples[:,i-1].eq(self.tgt_pad).sum() == num_samples:
                    break

                # eos_mask = out_indexes.ne(tgt_pad)
                # pre_eos_mask = samples[:,i].ne(tgt_eos).ne(tgt_pad)
                _lens = 1 - samples[:,i-1].eq(self.tgt_eos) - samples[:,i-1].eq(self.tgt_pad)
                lengths = lengths + _lens.type_as(lengths)

                pad_mask = samples[:, i-1].eq(self.tgt_pad) + samples[:,i-1].eq(self.tgt_eos)
                # samples[:,i] = out_indexes.masked_fill_(pad_mask, tgt_pad)
                out_indexes = out_indexes.squeeze(1)
                out_indexes.masked_fill_(pad_mask, self.tgt_pad)
                samples[:, i] = out_indexes

        else:
            for i in range(given_len):
                out, h = self.forward(samples[:, i], h, context, mask_src)
                # out, h, att_i = self.forward(samples[:, i], h, context, mask_src)
                if way == 'random':
                # random sampling
                    out_indexes = torch.multinomial(torch.exp(out), 1)
                if way == 'greedy':
                # greedy sampling
                    out_indexes = torch.exp(out).max(1)[1].unsqueeze(1)


            pad_mask = samples[:, i].eq(self.tgt_pad) + samples[:, i].eq(self.tgt_eos)
            # samples[:,i] = out_indexes.masked_fill_(pad_mask, tgt_pad)
            out_indexes = out_indexes.squeeze(1)
            out_indexes.masked_fill_(pad_mask, self.tgt_pad)
            samples[:, i+1] = out_indexes

            _lens = 1 - samples[:, i].eq(self.tgt_eos) - samples[:, i].eq(self.tgt_pad)
            lengths = lengths + _lens.type_as(lengths)

            for i in range(given_len + 1, self.max_seq_len):
                out, h = self.forward(samples[:, i - 1], h, context, mask_src)

                if way == 'random':
                # random sampling
                    out_indexes = torch.multinomial(torch.exp(out), 1)
                if way == 'greedy':
                # greedy sampling
                    out_indexes = torch.exp(out).max(1)[1].unsqueeze(1)

                if samples[:, i - 1].eq(self.tgt_pad).sum() == num_samples:
                    break
                # eos_mask = out_indexes.ne(tgt_pad)
                # pre_eos_mask = samples[:,i].ne(tgt_eos).ne(tgt_pad)
                _lens = 1 - samples[:, i - 1].eq(self.tgt_eos) - samples[:, i - 1].eq(self.tgt_pad)
                lengths = lengths + _lens.type_as(lengths)    # todo: length is not correct

                pad_mask = samples[:, i - 1].eq(self.tgt_pad) + samples[:, i - 1].eq(self.tgt_eos)
                # samples[:,i] = out_indexes.masked_fill_(pad_mask, tgt_pad)
                out_indexes = out_indexes.squeeze(1)
                out_indexes.masked_fill_(pad_mask, self.tgt_pad)
                samples[:, i] = out_indexes

        return samples, lengths

    def batchNLLLoss(self, src_input, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """
        weight = torch.ones(self.tgt_vocab_size).cuda()
        weight[self.tgt_pad] = 0
        loss_fn = nn.NLLLoss(weight, reduction='sum') #损失函数按权重相加，并且不平均

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        mask_src = src_input[0].eq(self.src_pad).permute(1, 0)

        context, enc_hidden = self.encoder(src_input)
        # hi=[-> hi; <- hi]
        h =torch.cat((enc_hidden[0], enc_hidden[1]), dim=-1).unsqueeze(0)
        # h1 = torch.cat((enc_hidden[2], enc_hidden[3]), dim=-1).unsqueeze(0)
        # h = torch.cat((h0, h1), dim=0)
        """
        if self.gpu:
            inp = inp.cuda()
            h = h.cuda()
            target = target.cuda()
        """
        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h, context, mask_src)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, src_input, inp, target, rewards):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        mask_src = src_input[0].eq(self.src_pad).permute(1, 0)
        context, enc_hidden = self.encoder(src_input)
        # hi=[-> hi; <- hi]
        h = torch.cat((enc_hidden[0],enc_hidden[1]), dim=-1).unsqueeze(0)
        # h1 = torch.cat((enc_hidden[2], enc_hidden[3]), dim=-1).unsqueeze(0)
        # h = torch.cat((h0, h1), dim=0)
        loss = 0

        for i in range(seq_len):
            out, h = self.forward(inp[i], h, context, mask_src)
            mask = inp[i].eq(self.tgt_eos).float() + inp[i].eq(self.tgt_pad).float()
            mask = 1 - mask
            if any(mask):
                rewards_i = rewards[i+1] * mask
                # TODO: should h be detached from graph (.detach())?
                for j in range(batch_size):
                    loss += -out[j][target.data[i][j]] * rewards_i[j]  # log(P(y_t|Y_1:Y_{t-1})) * Q
            else:
                break
        return loss / batch_size