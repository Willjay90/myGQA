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


class GQADataSet(Dataset):
    def __init__(self, input_file, root_dir, transform=None):
        self.spacy_en = spacy.load('en')

        self.input_file = input_file
        self.root_dir = root_dir
        self.transform = transform
        self.obj_info_dir = root_dir + "/allImages/objects/gqa_objects_info.json"
        self.spatial_info_dir = root_dir + "/allImages/spatial/gqa_spatial_info.json"

        with open(input_file, 'r') as f:
            self.questions = json.load(f)
        self.quesiotn_key_list = list(self.questions.keys())

        with open(self.obj_info_dir, 'r') as f:
            self.obj_info = json.load(f)

        with open(self.spatial_info_dir, 'r') as f:
            self.spatial_info = json.load(f)

    def __len__(self):
        return len(self.questions)
    
    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]


    def __getitem__(self, idx):
        obj_idx = str(self.obj_info[self.questions[self.quesiotn_key_list[idx]]['imageId']]["file"])
        obj_dir = self.root_dir + "/allImages/objects/gqa_objects_" + obj_idx + ".h5"
        spatial_dir = self.root_dir + "/allImages/spatial/gqa_spatial_" + obj_idx + ".h5"

        obj_file = h5py.File(obj_dir, 'r')
        spatial_file = h5py.File(spatial_dir, 'r')

        obj_id = self.obj_info[self.questions[self.quesiotn_key_list[idx]]['imageId']]["idx"]
  
        spatial_features = spatial_file['features'][obj_id] # 2048 * 7 * 7
        obj_features = obj_file['features'][obj_id]         # 100 * 2048
        obj_bboxes = obj_file['bboxes'][obj_id]             # 100 * 4

        has_answer = 'answer' in self.questions[self.quesiotn_key_list[idx]]

        # text preprocessing
        src = self.tokenize_en(self.questions[self.quesiotn_key_list[idx]]['question'])
        tgt = self.tokenize_en(self.questions[self.quesiotn_key_list[idx]]['answer'] if has_answer else None)
        src_mask = (src != 0).unsqueeze(-2)
        print(src_mask)

        sample = {'question': self.questions[self.quesiotn_key_list[idx]]['question'],
                  'has_answer': has_answer,
                  'answer': self.questions[self.quesiotn_key_list[idx]]['answer'] if has_answer else None,
                  'image_feature': spatial_features,
                  'obj_features': obj_features,
                  'obj_bboxes': obj_bboxes,
                  'src': src,
                  'tgt': tgt
                 }

        if self.transform:
            sample = self.transform(sample)

        return sample