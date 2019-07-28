import os
from config.collections import AttrDict
from config.function_config_lib import ModelParPair

__C = AttrDict()
cfg = __C

__C.run = "train+predict"
__C.test_mode = False
# __C.test_mode = True


__C.num_workers = 6
__C.batch_size = 256
__C.question_len = 30
__C.image_feature_dim = 2048 * 7 * 7 
__C.answer_len = 30

# output
__C.out_dir = ''
__C.seed = 1400


__C.vocab_question_file = "data/gqa_question_vocab.txt"
__C.vocab_answer_file = "data/gqa_answer_vocab.txt"


# --------------------------------------------------------------------------- #
# training_parameters options:
# --------------------------------------------------------------------------- #
__C.training_parameters = AttrDict()
__C.training_parameters.report_interval = 100
__C.training_parameters.snapshot_interval = 1000
__C.training_parameters.max_iter = 60000
__C.training_parameters.clip_norm_mode = 'all'
__C.training_parameters.max_grad_l2_norm = 0.25
__C.training_parameters.wu_factor = 0.2
__C.training_parameters.wu_iters = 1000
__C.training_parameters.lr_steps = [5000, 7000, 9000, 11000]
__C.training_parameters.lr_ratio = 0.1


# --------------------------------------------------------------------------- #
# optimizer options:
# --------------------------------------------------------------------------- #
__C.optimizer = ModelParPair('Adamax')
