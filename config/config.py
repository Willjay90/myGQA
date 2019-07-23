import os
from config.collections import AttrDict
from config.function_config_lib import ModelParPair

__C = AttrDict()
cfg = __C

__C.test_mode = True
__C.num_workers = 6
__C.batch_size = 64
__C.max_len = 30
__C.image_feature_dim = 2048 * 7 * 7 


__C.optimizer = ModelParPair('Adamax')

__C.vocab_question_file = "data/gqa_question_vocab.txt"
__C.vocab_answer_file = "data/gqa_answer_vocab.txt"
