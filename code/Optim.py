import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import myAdam


class Optim(object):
    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'myadam':
            self.optimizer = myAdam.MyAdam(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)

        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, lr_decay=None, start_decay_at=None, max_weight_value=None, max_grad_norm=None):
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.max_weight_value = max_weight_value
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.best_metric = 0
        self.bad_count = 0
        self.decay_bad_count = 3

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        if self.max_weight_value:
            for p in self.params:
                p.data.clamp_(0 - self.max_weight_value, self.max_weight_value)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _updateLearningRate(self, epoch):
        if self.start_decay_at is not None and self.start_decay_at == epoch:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

    def updateLearningRate(self, bleu):
        # if self.start_decay_at is not None and epoch >= self.start_decay_at:
        #     self.start_decay = True
        # if self.last_ppl is not None and ppl > self.last_ppl:
        #     self.start_decay = True
        #
        # if self.start_decay:
        #     self.lr = self.lr * self.lr_decay
        #     print("Decaying learning rate to %g" % self.lr)

        # self.last_ppl = ppl
        if bleu >= self.best_metric:
            self.best_metric = bleu
            self.bad_count = 0
        else:
            self.bad_count += 1

        if self.bad_count >= self.decay_bad_count and self.lr >= 1e-4:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
            self.bad_count = 0
        self.optimizer.param_groups[0]['lr'] = self.lr

    def reset_learningrate(self, learningrate):
        self.lr = learningrate
        self.best_metric = 0
        print("Resetting learning rate to %g" % self.lr)
        self.optimizer.param_groups[0]['lr'] = self.lr
