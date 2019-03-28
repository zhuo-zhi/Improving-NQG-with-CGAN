from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator, Iterator
import torch
from config import tgt_vocab_size, src_vocab_size, batch_size, data_name


class ReversibleField(Field):
    def __init__(self, **kwargs):
        if kwargs.get('tokenize') is list:
            self.use_revtok = False
        else:
            self.use_revtok = True
        if kwargs.get('tokenize') is None:
            kwargs['tokenize'] = 'revtok'
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = ' UNK '
        super(ReversibleField, self).__init__(**kwargs)

    def reverse(self, batch, src_data=None, att=None, dic_src=None):
        if self.use_revtok:
            try:
                import revtok
            except ImportError:
                print("Please install revtok.")
                raise
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        if att is not None:
            for i in range(len(batch)):
                for j in range(len(batch[i])):
                    if batch[i][j] == '<unk>':
                        _, maxIndex = att[j].max(0)
                        batch[i][j] = dic_src.vocab.itos[src_data[maxIndex[0]]]

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w+'  ')
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        if self.use_revtok:
            return [revtok.detokenize(ex) for ex in batch]
        return [''.join(ex) for ex in batch]


def train_data():
    tokenize = lambda x: x.split()

    Text_src = Field(sequential=True, tokenize=tokenize, eos_token='<EOS>', include_lengths=True, lower=True)
    Answer = Field(sequential=True, tokenize=tokenize, eos_token='<EOS>', include_lengths=True, lower=True)
    Text_tgt = Field(sequential=True, tokenize=tokenize, eos_token='<EOS>',
                     include_lengths=True, init_token='<SOS>', lower=True)


    trn_datafields = [("source",Text_src),
                    ("target", Text_tgt),
                    ("answer", Answer)]
    trn, val = TabularDataset.splits(
        path="../data/"+str(data_name), # the root directory where the data lies
        train='train.json', validation = 'validation.json',
        format='json',
        # skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields={'source': trn_datafields[0], 'target': trn_datafields[1], 'answer': trn_datafields[2]})

    # Text_src.build_vocab(trn, max_size=vocab_size)
    Text_src.build_vocab(trn, max_size=src_vocab_size)
    Text_tgt.build_vocab(trn, max_size=tgt_vocab_size)
    Answer.build_vocab(trn)
    Text_src.vocab.load_vectors("glove.840B.300d")
    Text_tgt.vocab.load_vectors("glove.840B.300d")

    train_iter, val_iter = BucketIterator.splits(
            (trn, val), # we pass in the datasets we want the iterator to draw data from
            batch_sizes= (batch_size, batch_size),
            device=-1, # if you want to use the GPU, specify the GPU number here
            sort_key=lambda x: len(x.source), # the BucketIterator needs to be told what function it should use to group the data.
            sort_within_batch=True,
            shuffle = True,
            repeat= False)


    Text_tgt_r = ReversibleField(sequential=True, include_lengths=True,
                                 eos_token='<EOS>', init_token='<SOS>', lower=True)
    Text_tgt_r.vocab = Text_tgt.vocab

    Text_src_r = ReversibleField(sequential=True, include_lengths=True,
                                 eos_token='<EOS>', lower=True)
    Text_src_r.vocab = Text_src.vocab

    Text_ans_r = ReversibleField(sequential=True, tokenize=tokenize,
                   eos_token='<EOS>', include_lengths=True, lower=True)
    Text_ans_r.vocab = Answer.vocab

    src_pad = Text_src.vocab.stoi['<pad>']
    src_unk = Text_src.vocab.stoi['<unk>']
    src_eos = Text_src.vocab.stoi['<EOS>']
    src_special = [src_pad, src_unk, src_eos]

    ans_pad = Answer.vocab.stoi['<pad>']
    ans_unk = Answer.vocab.stoi['<unk>']
    ans_eos = Answer.vocab.stoi['<EOS>']
    ans_special = [ans_pad, ans_unk, ans_eos]

    tgt_pad = Text_tgt.vocab.stoi['<pad>']
    tgt_unk = Text_tgt.vocab.stoi['<unk>']
    tgt_eos = Text_tgt.vocab.stoi['<EOS>']
    tgt_sos = Text_tgt.vocab.stoi['<SOS>']
    tgt_special = [tgt_pad, tgt_unk, tgt_eos, tgt_sos]


    # discriminator data iterator
    passage = Field(sequential=True, tokenize=tokenize, eos_token='<EOS>', include_lengths=True, lower=True)
    ans = Field(sequential=True, tokenize=tokenize, eos_token='<EOS>', include_lengths=True, lower=True)
    ques = Field(sequential=True, tokenize=tokenize, eos_token='<EOS>',include_lengths=True, lower=True)
    target = Field(sequential=False, use_vocab=False)

    disc_trn_datafields = [("question", ques),
                      ("answer", ans),
                      ("passage", passage),
                      ("target", target)]

    disc_trn = TabularDataset(
        path="../data/" + str(data_name) + "/disc.json",  # the root directory where the data lies
        # train='disc.json',
        format='json',
        # skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields={'question': disc_trn_datafields[0], 'answer': disc_trn_datafields[1], 'passage': disc_trn_datafields[2], 'target': disc_trn_datafields[3]})

    passage.vocab = Text_src.vocab
    ans.vocab = Answer.vocab
    ques.vocab = Text_tgt.vocab

    disc_train_iter = BucketIterator(
        dataset=disc_trn,  # we pass in the datasets we want the iterator to draw data from
        batch_size = batch_size,
        device=-1,  # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: len(x.question),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        shuffle=True,
        repeat=False)



    # raw data iterator
    Text_tgt_raw = ReversibleField(sequential=True, tokenize=tokenize, include_lengths=True, lower=True)

    trn_datafields = [("source", Text_tgt_raw),
                      ("target", Text_tgt_raw)]
    trn_raw, val_raw = TabularDataset.splits(
        path="../data/"+str(data_name),  # the root directory where the data lies
        train='train.json', validation='validation.json',
        format='json',
        # skip_header=True,
        # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields={'source': trn_datafields[0], 'target': trn_datafields[1]})


    Text_tgt_raw.build_vocab(val_raw)

    train_iter_raw, val_iter_raw = BucketIterator.splits(
        (trn_raw, val_raw),  # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(batch_size, batch_size),
        device=-1,  # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: len(x.source),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        shuffle=True,
        repeat=False)


    return train_iter, val_iter, src_special, tgt_special, Text_tgt_r, val_iter_raw, Text_tgt_raw, Text_src_r,\
           Text_src, Text_tgt, ans_special, Text_ans_r, disc_train_iter

