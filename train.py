import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import time
from dataset_utils.dataset_utils import prepare_train_data_set, \
    prepare_eval_data_set, prepare_test_data_set
from train_model.Engineer import one_stage_train
from train_model.model_factory import make_model
import numpy as np
from dataset_utils import text_processing
import torch.nn.functional as F
from config.config import cfg
import torch.optim as optim
import os
from config.config_utils import finalize_config, dump_config
from train_model.Engineer import one_stage_train, compute_a_batch
from train_model.helper import run_model, print_result
from bisect import bisect

def print_eval(prepare_data_fun, out_label):
    model_file = os.path.join(snapshot_dir, "best_model.pth")
    pkl_res_file = os.path.join(snapshot_dir,
                                "best_model_predict_%s.pkl" % out_label)
    out_file = os.path.join(snapshot_dir,
                            "best_model_predict_%s.json" % out_label)

    data_set_test = prepare_test_data_set('data', **cfg)

    data_reader_test = DataLoader(dataset=data_set_test,
                                 batch_size=cfg.batch_size,
                                 shuffle=False,
                                 num_workers=cfg.num_workers)

    vocab_dict = text_processing.VocabDict(cfg.vocab_question_file)
    ans_dic = text_processing.VocabDict(cfg.vocab_answer_file)


    my_model = make_model(vocab_dict.num_vocab, ans_dic.num_vocab, cfg.image_feature_dim)
    model.load_state_dict(torch.load(model_file)['state_dict'])
    model.eval()

    question_ids, soft_max_result = run_model(model, data_reader_test)

    print_result(question_ids,
                 soft_max_result,
                 ans_dic,
                 out_file,
                 json_only=False,
                 pkl_res_file=pkl_res_file)

def lr_lambda_fun(i_iter):
    if i_iter <= cfg.training_parameters.wu_iters:
        alpha = float(i_iter) / float(cfg.training_parameters.wu_iters)
        return cfg.training_parameters.wu_factor * (1. - alpha) + alpha
    else:
        idx = bisect(cfg.training_parameters.lr_steps, i_iter)
        return pow(cfg.training_parameters.lr_ratio, idx)

def get_optim_scheduler(optimizer):
    return LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

def run_epoch(data_iter, model, optimizer, loss_criterion):
    start = time.time()
    avg_accuracy = 0
    accuracy_decay = 0.99

    for i, batch in enumerate(data_iter):    
        optimizer.zero_grad()
        add_graph = False
        scores, total_loss, n_sample = compute_a_batch(batch, model, eval_mode=False,
                                                        loss_criterion=loss_criterion,
                                                        add_graph=add_graph)
        total_loss.backward()
        accuracy = scores.item() / n_sample
        avg_accuracy += (1 - accuracy_decay) * (accuracy - avg_accuracy)
        optimizer.step()

        if i % 100 == 0:
            print("iteration: ", i, " loss: ", total_loss.item(), " acc: ", avg_accuracy)

        
if __name__ == '__main__':

    seed = cfg.seed
    out_dir = cfg.out_dir if cfg.out_dir is not None else os.getcwd()
    snapshot_dir = os.path.join(out_dir, "results", str(seed))
    boards_dir = os.path.join(out_dir, "boards", str(seed))

    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    if not os.path.exists(boards_dir):
        os.makedirs(boards_dir)

    # dump the config file to snap_shot_dir
    config_to_write = os.path.join(snapshot_dir, "config.yaml")
    dump_config(cfg, config_to_write)

    data_set_trn = prepare_train_data_set('data', **cfg)
    data_set_val = prepare_eval_data_set('data', **cfg)

    data_reader_trn = DataLoader(dataset=data_set_trn,
                                 batch_size=cfg.batch_size,
                                 shuffle=True,
                                 num_workers=cfg.num_workers)
    data_reader_val = DataLoader(data_set_val,
                                 shuffle=True,
                                 batch_size=cfg.batch_size,
                                 num_workers=cfg.num_workers)
    
    vocab_dict = text_processing.VocabDict(cfg.vocab_question_file)
    answer_dict = text_processing.VocabDict(cfg.vocab_answer_file)

    my_model = make_model(vocab_dict.num_vocab, answer_dict.num_vocab, cfg.image_feature_dim)

    if hasattr(my_model, 'module'):
        model = my_model.module

    params = model.parameters()
    optimizer = getattr(optim, cfg.optimizer.method)(params, **cfg.optimizer.par)
    scheduler = get_optim_scheduler(optimizer)

    optimizer = optim.Adamax(model.parameters(), lr=2.5e-4)
    criterion = nn.BCEWithLogitsLoss()

    i_epoch = 0
    i_iter = 0
    best_accuracy = 0

    print("BEGIN TRAINING...")
    one_stage_train(model,
                data_reader_trn,
                optimizer, criterion, data_reader_eval=data_reader_val,
                snapshot_dir=snapshot_dir, log_dir=boards_dir,
                start_epoch=i_epoch, i_iter=i_iter,
                scheduler=scheduler, best_val_accuracy=best_accuracy)
    print("END TRAINING...")

    print("BEGIN PREDICTING ON TEST/VAL set...")

    if 'predict' in cfg.run:
        print_eval(prepare_test_data_set, "test")
    if cfg.run == 'train+val':
        print_eval(prepare_eval_data_set, "val")


    # dataiter = iter(data_reader_trn)
    # dataiter.next()
    
    # for epoch in range(1):
    #     run_epoch(data_reader_trn, model, optimizer, criterion)
