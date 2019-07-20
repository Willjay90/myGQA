import json
import argparse
import os
import re
import h5py
from dataset_utils.gqa_dataset import GQADataSet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files",
                        type=str,
                        required=True,
                        nargs='+',
                        help="input test json file")

    args = parser.parse_args()
    data_dir = args.input_files[0]

    # question_dir = data_dir + "/questions1.2/testdev_all_questions.json"
    question_dir = data_dir + "/questions1.2/submission_all_questions.json"

    
    object_dir = data_dir + "/allImages/objects/gqa_objects_info.json"

    with open(question_dir, 'r') as f:
        data = json.load(f)
    key_list = list(data.keys())
    print('question' in data[key_list[0]])
    print('answer' in data[key_list[0]])

    print(data[key_list[0]]['imageId'])     # n315887
    print(data[key_list[0]]['question'])    # Is the mouse to the right of the keyboard both smooth and gray?
    # print(data[key_list[0]]['answer'])    # yes

    with open(object_dir, 'r') as f:
        obj_data = json.load(f)
    print(obj_data[data[key_list[0]]['imageId']]) # {'idx': 6858, 'file': 11, 'objectsNum': 27, 'width': 640, 'height': 425}

    object_dir = data_dir + "/allImages/objects/gqa_objects_" + str(obj_data[data[key_list[0]]['imageId']]["file"]) + ".h5"

    obj_file = h5py.File(object_dir, 'r')
    
    print(list(obj_file.keys()))            # ['bboxes', 'features']
    print(obj_file['features'][6858].shape) # 100 * 2048
    print(obj_file['bboxes'][6858].shape)   # 100 * 4

    gqa_dataset = GQADataSet(input_file=question_dir, root_dir=data_dir)
    for i in range(len(gqa_dataset)):
        sample = gqa_dataset[i]
        print(i, sample['question'], sample['has_answer'], sample['obj_features'].shape)

    vocab_question_file = "vocabulary_gqa.txt"
    vocab_answer_file = "answers_gqa.txt"