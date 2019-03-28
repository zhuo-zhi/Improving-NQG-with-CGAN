from __future__ import division
import torch


class Beam(object):
    def __init__(self, size, tgt_special, cuda=False):
        self.pad = tgt_special[0]
        self.unk = tgt_special[1]
        self.eos = tgt_special[2]
        self.sos = tgt_special[3]

        self.size = size
        self.done = False

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.all_length = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.sos
        self.nextYs_true = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs_true[0][0] = self.sos

        # The attentions (matrix) for each time.
        self.attn = []


    # Get the outputs for the current timestep.
    def getCurrentState(self):
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def getCurrentOrigin(self):
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.
    def advance(self, wordLk, attnOut):
        numAll = wordLk.size(1)
        allScores = wordLk

        # self.length += 1  # TODO: some is finished so do not acc length for them
        if len(self.prevKs) > 0:
            finish_index = self.nextYs[-1].eq(self.eos)
            if any(finish_index):
                # wordLk.masked_fill_(finish_index.unsqueeze(1).expand_as(wordLk), -float('inf'))
                allScores.masked_fill_(finish_index.unsqueeze(1).expand_as(allScores), -float('inf'))
                for i in range(self.size):
                    if self.nextYs[-1][i] == self.eos:
                        # wordLk[i][s2s.Constants.EOS] = 0
                        allScores[i][self.eos] = 0
            # set up the current step length
            cur_length = self.all_length[-1]
            for i in range(self.size):
                cur_length[i] += 0 if self.nextYs[-1][i] == self.eos else 1

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            prev_score = self.all_scores[-1]
            # now_acc_score = wordLk + prev_score.unsqueeze(1).expand_as(wordLk)
            # beamLk = now_acc_score / cur_length.unsqueeze(1).expand_as(now_acc_score)
            now_acc_score = allScores + prev_score.unsqueeze(1).expand_as(allScores) # 5 * 20088
            beamLk = now_acc_score / cur_length.unsqueeze(1).expand_as(now_acc_score)
        else:
            self.all_length.append(self.tt.FloatTensor(self.size).fill_(1))
            # beamLk = wordLk[0]
            beamLk = allScores[0]  # 20098

        flatBeamLk = beamLk.view(-1)    # 20098  ,  5*20098

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numAll
        # predict = bestScoresId - prevK * numWords
        predict = bestScoresId - prevK * numAll

        if len(self.prevKs) > 0:
            self.all_length.append(cur_length.index_select(0, prevK))
            self.all_scores.append(now_acc_score.view(-1).index_select(0, bestScoresId))
        else:
            self.all_scores.append(self.scores)

        self.prevKs.append(prevK)
        self.nextYs.append(predict)
        self.attn.append(attnOut.index_select(0, prevK))  # attnOUt : 5 * 98

        # End condition is when every one is EOS.
        if all(self.nextYs[-1].eq(self.eos)):
            self.done = True

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def getBest(self):
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def getHyp(self, k):
        hyp, attn = [], []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
            attn.append(self.attn[j][k])

        return hyp[::-1], torch.stack(attn[::-1])
