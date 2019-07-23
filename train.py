import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import time
from dataset_utils.dataset_utils import prepare_train_data_set, \
    prepare_eval_data_set, prepare_test_data_set
from train_model.Engineer import one_stage_train
from train_model.model_factory import make_model
import numpy as np
from dataset_utils import text_processing
import torch.nn.functional as F
from config.config import cfg


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squezze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
        
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1) / norm)
        loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm

class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # send out to different GPUs
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)

        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, requires_grad=self.opt is not None)] for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)), t[:, i:i+chunk_size].contiguous().view(-1)) for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])

            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    # if torch.cuda.is_available():
    #   np_mask = np_mask.cuda()
    return np_mask

def create_mask(src, trg, pad=0):
    src_mask = (src != pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        # if torch.cuda.is_available():
        #     np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
        trg_y = trg[:, 1:]
        ntokens = (trg_y != pad).data.sum()

    else:
        trg_mask = None
    return src_mask, trg_mask,ntokens

def cal_performance(pred, gold, smoothing=False):
    "Apply label smoothing if needed"
    print("pred: ", pred.size(), "gold: ", gold.size())
    loss = cal_loss(pred, gold, smoothing)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(0)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    # gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        print("pred", pred.size(), "gold", gold.size())
        loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')

    return loss


def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for i, batch in enumerate(data_iter):        
        src = batch['question_seq']
        trg = batch['answer_seq']
        img_feature = batch['image_feature']

        # if torch.cuda.is_available():
        #     src = src.cuda()
        #     trg = trg.cuda()
        #     img_feature = img_feature.cuda()

        src_mask, trg_mask, ntokens = create_mask(src, trg)
        
        pred = model.forward(src, trg, img_feature, src_mask, trg_mask)

        loss = loss_compute(pred, trg, ntokens)
        print(loss)

        # total_loss += loss
        # print(total_loss)
    #     total_tokens += batch.ntokens
    #     tokens += batch.tokens

    #     if i % 10 == 1:
    #         elapsed = time.time() - start
    #         print("Epoch Step: %d Loss: %f Tokens per sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
    #         start = time.time()
    #         tokens = 0
    # return total_loss / total_tokens

if __name__ == '__main__':

    data_set_trn = prepare_train_data_set('data', **cfg)
    # data_set_val = prepare_train_data_set('data', **cfg)
    # data_set_test = prepare_train_data_set('data', **cfg)

    data_reader_trn = DataLoader(dataset=data_set_trn,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=cfg.num_workers)
    # data_reader_val = DataLoader(data_set_val,
    #                              shuffle=True,
    #                              batch_size=512,
    #                              num_workers=6)

    vocab_dict = text_processing.VocabDict(cfg.vocab_question_file)
    answer_dict = text_processing.VocabDict(cfg.vocab_answer_file)

    model = make_model(vocab_dict.num_vocab, answer_dict.num_vocab, cfg.image_feature_dim)

    # GPUs
    devices = [0, 1, 2, 3]

    criterion = LabelSmoothing(size=cfg.max_len, padding_idx=0, smoothing=0.0)

    if torch.cuda.device_count() > 1:
        model = model.module
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) 

    for i in range(1):
        model.train()
        # run_epoch(data_reader_trn, model, MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))
        run_epoch(data_reader_trn, model, SimpleLossCompute(model.generator, criterion, opt=model_opt))
