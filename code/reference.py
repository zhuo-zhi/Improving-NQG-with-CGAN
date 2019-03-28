from collections import defaultdict

def ref(src_rev, rev, rev_raw, train_iter, val_iter_raw):
    def delete_pad(sent):
        pad = '<pad>'
        i = len(sent) - 1
        while sent[i] == pad:
            del sent[i]
            i -=1
        return sent

    train_ref = defaultdict(list)
    for i, data in enumerate(train_iter):
        # src_data = data.source[0].permute(1, 0)  # batch_size, length
        src_data = data.source[0]
        src_data = src_rev.reverse(src_data)
        tgt_data = data.target[0]
        tgt = rev.reverse(tgt_data)  # todo: reverse 有 unk
        batch_size = len(src_data)
        for k in range(batch_size):
            # key = " ".join([str(idx.item()) for idx in src_data[k]])
            key = " ".join([idx for idx in src_data[k]])
            train_ref[key].append(tgt[k].split())


    val_ref = defaultdict(list)
    val_reference = []
    for i, data in enumerate(val_iter_raw):
        src_data = data.source[0].permute(1, 0)  # batch_size, length
        tgt_data = data.target[0]
        tgt = rev_raw.reverse(tgt_data)
        batch_size = src_data.size(0)
        for k in range(batch_size):
            key = " ".join([str(idx.item()) for idx in src_data[k]])
            val_ref[key].append(delete_pad(tgt[k].split()))

    for i, data in enumerate(val_iter_raw):
        src_data = data.source[0].permute(1, 0)  # batch_size, length
        batch_size = src_data.size(0)
        for k in range(batch_size):
            key = " ".join([str(idx.item()) for idx in src_data[k]])
            val_reference.append(val_ref[key])

    # f = open('ref.txt', 'w', encoding='utf-8')
    # f.write(str(train_reference))
    # f.close()
    return  train_ref, val_reference

if __name__ == '__main__':
    ref(src_rev, rev, rev_raw, train_ref_iter, val_iter_raw)