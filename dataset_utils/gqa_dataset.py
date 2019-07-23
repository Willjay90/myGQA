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
        self.max_len = data_params['max_len']

        self.input_file = input_file
        self.root_dir = root_dir
        self.transform = transform
        self.obj_info_dir = root_dir + "/allImages/objects/gqa_objects_info.json"
        self.spatial_info_dir = root_dir + "/allImages/spatial/gqa_spatial_info.json"

        self.vocab_dict = text_processing.VocabDict(data_params['vocab_question_file'])
        self.answer_dict = text_processing.VocabDict(data_params['vocab_answer_file'])

        with open(input_file, 'r') as f:
            self.questions = json.load(f)
        self.quesiotn_key_list = list(self.questions.keys())

        with open(self.obj_info_dir, 'r') as f:
            self.obj_info = json.load(f)

        with open(self.spatial_info_dir, 'r') as f:
            self.spatial_info = json.load(f)

    def __len__(self):
        if self.test_mode:
            return 10
        return len(self.questions)

    def _get_image_features_(self, idx):
        obj_idx = str(self.obj_info[self.questions[self.quesiotn_key_list[idx]]['imageId']]["file"])
        obj_dir = self.root_dir + "/allImages/objects/gqa_objects_" + obj_idx + ".h5"
        spatial_dir = self.root_dir + "/allImages/spatial/gqa_spatial_" + obj_idx + ".h5"

        obj_file = h5py.File(obj_dir, 'r')
        spatial_file = h5py.File(spatial_dir, 'r')

        obj_id = self.obj_info[self.questions[self.quesiotn_key_list[idx]]['imageId']]["idx"]
  
        spatial_feature = spatial_file['features'][obj_id]  # 2048 * 7 * 7
        obj_feature = obj_file['features'][obj_id]          # 100 * 2048
        obj_bboxes = obj_file['bboxes'][obj_id]             # 100 * 4
        return spatial_feature, obj_feature, obj_bboxes
 
    def __getitem__(self, idx):
        sample = dict()

        # preprocess question token
        question = self.questions[self.quesiotn_key_list[idx]]['question'].lower()
        question_tokens = text_processing.tokenize(question)

        input_seq = np.zeros((self.max_len), np.long)
        question_inds = ([self.vocab_dict.word2idx(w) for w in question_tokens])
        seq_length = len(question_inds)
        read_len = min(seq_length, self.max_len)
        input_seq[:read_len] = question_inds[:read_len]

        sample['question'] = question
        sample['question_seq'] = input_seq

        # preprocess answer token
        has_answer = 'answer' in self.questions[self.quesiotn_key_list[idx]]
        sample['has_answer'] = has_answer

        if has_answer:
            answer = self.questions[self.quesiotn_key_list[idx]]['answer'].lower()
            answer_tokens = text_processing.tokenize(answer)
            answer_seq = np.zeros((3), np.long) # data preprocessing result (answer's max len is 3)
            if len(answer_tokens) == 1:
                ans_idx = [self.answer_dict.word2idx(answer)]
            else:
                ans_idx = ([self.answer_dict.word2idx(ans) for ans in answer])
            seq_length = len(ans_idx)
            read_len = min(seq_length, 3)
            answer_seq[:read_len] = ans_idx[:read_len]

            sample['answer'] = answer
            sample['answer_seq'] = answer_seq

        # image features
        spatial_feature, obj_feature, obj_bboxes = self._get_image_features_(idx)

        sample['image_feature'] = spatial_feature
        sample['obj_feature'] = obj_feature
        sample['obj_bboxes'] = obj_bboxes


        if self.transform:
            sample = self.transform(sample)

        return sample