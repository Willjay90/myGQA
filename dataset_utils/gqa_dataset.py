import os
import torch
import numpy as numpy
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
import argparse
import h5py
import spacy
import pandas as pd
import numpy as np
from dataset_utils import text_processing

class GQADataSet(Dataset):
    def __init__(self, input_file, root_dir, transform=None, **data_params):
        self.spacy_en = spacy.load('en')
        self.test_mode = data_params['test_mode']

        self.data_params = data_params
        self.question_len = data_params['question_len']
        self.answer_len = data_params['answer_len']


        self.input_file = input_file
        self.root_dir = root_dir
        self.transform = transform
        self.obj_info_dir = root_dir + "/allImages/objects/gqa_objects_info.json"
        self.spatial_info_dir = root_dir + "/allImages/spatial/gqa_spatial_info.json"

        self.vocab_dict = text_processing.VocabDict(data_params['vocab_question_file'])
        self.answer_dict = text_processing.VocabDict(data_params['vocab_answer_file'])

        with open(input_file, 'r') as f:
            self.questions = json.load(f)
        self.question_key_list = list(self.questions.keys())

        with open(self.obj_info_dir, 'r') as f:
            self.obj_info = json.load(f)

        with open(self.spatial_info_dir, 'r') as f:
            self.spatial_info = json.load(f)

    def __len__(self):
        if self.test_mode:
            return 2000
        return len(self.questions)

    def _get_image_features_(self, idx):
        obj_idx = str(self.obj_info[self.questions[self.question_key_list[idx]]['imageId']]["file"])
        obj_dir = self.root_dir + "/allImages/objects/gqa_objects_" + obj_idx + ".h5"
        spatial_dir = self.root_dir + "/allImages/spatial/gqa_spatial_" + obj_idx + ".h5"

        obj_file = h5py.File(obj_dir, 'r')
        spatial_file = h5py.File(spatial_dir, 'r')

        obj_id = self.obj_info[self.questions[self.question_key_list[idx]]['imageId']]["idx"]
  
        spatial_feature = spatial_file['features'][obj_id]  # 2048 * 7 * 7
        obj_feature = obj_file['features'][obj_id]          # 100 * 2048
        obj_bboxes = obj_file['bboxes'][obj_id]             # 100 * 4
        return spatial_feature, obj_feature, obj_bboxes
    
    def one_hot(self, target):
        dim = self.answer_dict.num_vocab
        return np.eye(dim, dtype=np.float32)[target]


    def __getitem__(self, idx):
        sample = dict()

        # preprocess question token
        question = self.questions[self.question_key_list[idx]]['question'].lower()
        question_tokens = text_processing.tokenize(question)

        input_seq = np.zeros((self.question_len), np.long)
        question_inds = ([self.vocab_dict.word2idx(w) for w in question_tokens])
        seq_length = len(question_inds)
        read_len = min(seq_length, self.question_len)
        input_seq[:read_len] = question_inds[:read_len]

        sample['question_id'] = self.question_key_list[idx]
        sample['question'] = question
        sample['question_seq'] = input_seq

        answer_seq = np.zeros((self.answer_len), np.long) 

        # preprocess answer token
        has_answer = True if 'answer' in self.questions[self.question_key_list[idx]] else False
        if has_answer:
            answer = self.questions[self.question_key_list[idx]]['answer'].lower()
            # sample['answer'] = answer
            full_answer = self.questions[self.question_key_list[idx]]['fullAnswer'].lower()
            
            answer_tokens = text_processing.tokenize(answer)
            # answer_seq = np.zeros((self.answer_len), np.long) 
            # if answer len > 1
            if len(answer_tokens) == 1:
                ans_idx = [self.answer_dict.word2idx(answer)]    
            else:
                ans_idx = ([self.answer_dict.word2idx(ans) for ans in answer])
            full_ans_idx = ([self.answer_dict.word2idx(ans) for ans in full_answer])

            seq_length = len(full_ans_idx)
            read_len = min(seq_length, self.answer_len)
            answer_seq[:read_len] = full_ans_idx[:read_len]
            # answer_seq[0] = ans_idx[0]
            # answer_seq[1] = 0
            # answer_seq[2:read_len] = full_ans_idx[:read_len-2]

            gold_seq = np.zeros((self.answer_len), np.long)
            gold_seq[:len(ans_idx)] = ans_idx[: len(ans_idx)]
            sample['answer'] = gold_seq
            sample['answer_gold'] = self.one_hot(gold_seq)

        sample['answer_seq'] = answer_seq

        # make 1/10 data without answer !!!! (test set does not have answer => make trg less important)
        if idx % 3 == 0:
            sample['answer_seq'] = np.zeros((self.answer_len), np.long) 


        # image features
        spatial_feature, obj_feature, obj_bboxes = self._get_image_features_(idx)

        sample['image_feature'] = spatial_feature
        sample['obj_feature'] = obj_feature
        sample['obj_bboxes'] = obj_bboxes


        if self.transform:
            sample = self.transform(sample)

        return sample