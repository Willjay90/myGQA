import torch
import torch.nn as nn
import sys
import os
import time
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from tools.timer import Timer
from config.config import cfg
from dataset_utils import text_processing

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if torch.cuda.is_available():
      np_mask = np_mask.cuda()
    return np_mask

def create_mask(src, trg, pad=0):
    src_mask = (src != pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if torch.cuda.is_available():
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
        trg_y = trg[:, 1:]
        ntokens = (trg_y != pad).data.sum()

    else:
        trg_mask = None
        ntokens = None
    return src_mask, trg_mask, ntokens
    
def compute_score_with_logits(logits, labels):
    # return logits.argmax(-1) == labels
    score = torch.zeros(labels.size(0))
    pred = logits.argmax(-1)
    # print("pred:", pred.shape, "labels", labels.shape)
    answer_dict = text_processing.VocabDict(cfg.vocab_answer_file)
    for idx, pred_idx in enumerate(pred):
        # print("pred", answer_dict.idx2word(pred_idx[0].item()), "ans", answer_dict.idx2word(labels[idx][0].item()))
        if pred_idx[0].item() == labels[idx][0].item():
            score[idx] = 1
            
    return score

def compute_a_batch(batch, myModel, eval_mode, loss_criterion, add_graph=False, log_dir=None):
    obs_res = batch['answer_gold']  # batch_size x len_of_ans x len_ans_dic
    # ys = batch['answer_seq']        # it's easier to compute the acc [ batch_size x len_of_ans ]
    ys = batch['answer']        # it's easier to compute the acc [ batch_size x len_of_ans ]

    obs_res = Variable(obs_res.type(torch.FloatTensor))

    if torch.cuda.is_available():
        obs_res = obs_res.cuda()
        ys = ys.cuda()

    n_sample = obs_res.size(0)
    logit_res = one_stage_run_model(batch, myModel, eval_mode, add_graph, log_dir)
    # print("question:", batch['question'], "answer: ", batch['answer'])
    predicted_scores = torch.sum(compute_score_with_logits(logit_res, ys.data))

    total_loss = None if loss_criterion is None else loss_criterion(logit_res, obs_res)

    return predicted_scores, total_loss, n_sample


def save_a_report(i_iter, train_loss, train_acc, train_avg_acc, report_timer, writer, data_reader_eval,
                  myModel, loss_criterion):
    val_batch = next(iter(data_reader_eval))
    val_score, val_loss, n_val_sample = compute_a_batch(val_batch, myModel, eval_mode=True, loss_criterion=loss_criterion)
    val_acc = val_score.item() / n_val_sample

    print("iter:", i_iter, "train_loss: %.6f" % train_loss, " train_score: %.4f" % train_acc,
          " avg_train_score: %.4f" % train_avg_acc, "val_score: %.4f" % val_acc,
          "val_loss: %.6f" % val_loss.data, "time(s): % s" % report_timer.end())

    sys.stdout.flush()
    report_timer.start()

    writer.add_scalar('train_loss', train_loss, i_iter)
    writer.add_scalar('train_score', train_acc, i_iter)
    writer.add_scalar('train_score_avg', train_avg_acc, i_iter)
    writer.add_scalar('val_score', val_score, i_iter)
    writer.add_scalar('val_loss', val_loss.data, i_iter)


    for name, param in myModel.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), i_iter)

def one_stage_eval_model(data_reader_eval, myModel, loss_criterion=None):
    score_tot = 0
    n_sample_tot = 0
    loss_tot = 0
    for idx, batch in enumerate(data_reader_eval):
        score, loss, n_sample = compute_a_batch(batch, myModel, eval_mode=True, loss_criterion=loss_criterion)
        score_tot += score.item()
        n_sample_tot += n_sample
        if loss is not None:
            loss_tot += loss.item() * n_sample
    return score_tot / n_sample_tot, loss_tot / n_sample_tot, n_sample_tot

def save_a_snapshot(snapshot_dir, i_iter, iepoch, myModel, my_optimizer, loss_criterion, best_val_accuracy,
                    best_epoch, best_iter, snapshot_timer, data_reader_eval):
    model_snapshot_file = os.path.join(snapshot_dir, "model_%08d.pth" % i_iter)
    model_result_file = os.path.join(snapshot_dir, "result_on_val.txt")

    save_dic = {
        'epoch': iepoch,
        'iter': i_iter,
        'state_dict': myModel.state_dict(),
        'optimizer': my_optimizer.state_dict()}

    if data_reader_eval is not None:
        val_accuracy, avg_loss, val_sample_tot = one_stage_eval_model(data_reader_eval, myModel,
                                                                      loss_criterion=loss_criterion)
        print("i_epoch:", iepoch, "i_iter:", i_iter, "val_loss:%.4f" % avg_loss,
              "val_acc:%.4f" % val_accuracy, "runtime: %s" % snapshot_timer.end())
        snapshot_timer.start()
        sys.stdout.flush()

        with open(model_result_file, 'a') as fid:
            fid.write('%d %d %.5f\n' % (iepoch, i_iter, val_accuracy))

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = iepoch
            best_iter = i_iter
            best_model_snapshot_file = os.path.join(snapshot_dir, "best_model.pth")

        save_dic['best_val_accuracy'] = best_val_accuracy
        torch.save(save_dic, model_snapshot_file)

        if best_iter == i_iter:
            if os.path.exists(best_model_snapshot_file):
                os.remove(best_model_snapshot_file)
            os.link(model_snapshot_file, best_model_snapshot_file)

    return best_val_accuracy, best_epoch, best_iter


def clip_gradients(myModel, i_iter, writer):
    max_grad_l2_norm = cfg.training_parameters.max_grad_l2_norm
    clip_norm_mode = cfg.training_parameters.clip_norm_mode
    if max_grad_l2_norm is not None:
        if clip_norm_mode == 'all':
            norm = nn.utils.clip_grad_norm(myModel.parameters(), max_grad_l2_norm)
            writer.add_scalar('grad_norm', norm, i_iter)
        elif clip_norm_mode == 'question':
            norm = nn.utils.clip_grad_norm(myModel.module.question_embedding_models.parameters(),
                                            max_grad_l2_norm)
            writer.add_scalar('question_grad_norm', norm, i_iter)
        else:
            raise NotImplementedError

def one_stage_run_model(batch, my_model, eval_mode, add_graph=False, log_dir=None):
    src = batch['question_seq']
    img_feature = batch['image_feature']
    answer_dict = text_processing.VocabDict(cfg.vocab_answer_file)

    if eval_mode:
        my_model.eval()
    else:
        my_model.train()
    
    trg = batch['answer_seq']

    if torch.cuda.is_available():
        src = src.cuda()
        img_feature = img_feature.cuda()
        trg = trg.cuda()

    src_mask, trg_mask, _ = create_mask(src, trg)

    logit_res = my_model(src, trg, img_feature, src_mask, trg_mask)

    return logit_res

def one_stage_train(myModel, data_reader_trn, my_optimizer,
                    loss_criterion, snapshot_dir, log_dir,
                    i_iter, start_epoch, best_val_accuracy=0, data_reader_eval=None,
                    scheduler=None):
    report_interval = cfg.training_parameters.report_interval
    snapshot_interval = cfg.training_parameters.snapshot_interval
    max_iter = cfg.training_parameters.max_iter

    avg_accuracy = 0
    accuracy_decay = 0.99
    best_epoch = 0
    writer = SummaryWriter(log_dir)
    best_iter = i_iter
    iepoch = start_epoch
    snapshot_timer = Timer('m')
    report_timer = Timer('s')

    while i_iter < max_iter:
        iepoch += 1
        for i, batch in enumerate(data_reader_trn):
            i_iter += 1
            if i_iter > max_iter:
                break

            scheduler.step(i_iter)

            my_optimizer.zero_grad()
            add_graph = False

            scores, total_loss, n_sample = compute_a_batch(batch, myModel, eval_mode=False,
                                                           loss_criterion=loss_criterion,
                                                           add_graph=add_graph, log_dir=log_dir)
            total_loss.backward()
            accuracy = scores.item() / n_sample
            avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)

            clip_gradients(myModel, i_iter, writer)
            my_optimizer.step()

            if i_iter % report_interval == 0:
                save_a_report(i_iter, total_loss.item(), accuracy, avg_accuracy, report_timer,
                              writer, data_reader_eval, myModel, loss_criterion)

            if i_iter % snapshot_interval == 0 or i_iter == max_iter:
                best_val_accuracy, best_epoch, best_iter = save_a_snapshot(snapshot_dir, i_iter, iepoch, myModel,
                                                                         my_optimizer, loss_criterion, best_val_accuracy,
                                                                          best_epoch, best_iter, snapshot_timer,
                                                                          data_reader_eval)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()
    print("best_acc:%.6f after epoch: %d/%d at iter %d" % (best_val_accuracy, best_epoch, iepoch, best_iter))
    sys.stdout.flush()
