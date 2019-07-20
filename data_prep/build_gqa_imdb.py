import numpy as np
import json
import os
from dataset_utils import text_processing
import argparse
from dataset_utils.create_imdb_header import create_header


def extract_answers(q_answers, valid_answer_set):
    all_answers = q_answers
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers


def build_imdb(image_set,
               valid_answer_set,
               coco_set_name=None,
               annotation_set_name=None):

    print('building imdb %s' % image_set)
    has_answer = False
    has_gt_layout = False
    load_gt_layout = False
    load_answer = False
    
    annotations = dict()
    image_name_template = 'GQA_' + image_set + '_%s'
    
    if image_set == 'train':
        question_file = os.path.join(data_dir, 'train_all_questions/train_all_questions_%s.json')
        load_answer = True
        
        for i in range(10):
            with open(question_file % i) as f:
                data = json.load(f)
                annotations.update(data)
    elif image_set == 'train_balanced':
        question_file = os.path.join(data_dir, 'train_balanced_questions.json')
        load_answer = True
        
        with open(question_file) as f:
            data = json.load(f)
            annotations.update(data)
    elif image_set == 'val':
        question_file = os.path.join(data_dir, 'val_all_questions.json')
        # question_file = os.path.join(data_dir, 'val_balanced_questions.json')
        load_answer = True
        
        with open(question_file) as f:
            data = json.load(f)
            annotations.update(data)
    else:
        question_file = os.path.join(data_dir, 'submission_all_questions.json')
        load_answer = False
        
        with open(question_file) as f:
            data = json.load(f)
            annotations.update(data)

    imdb = [None]*(len(annotations)+1) # 14305356 + 1 for GQA
    unk_ans_count = 0

    for n_q, key in enumerate(annotations):
        image_id = annotations[key]['imageId']
        question_id = key
        image_name = image_name_template % image_id

        feature_path = image_name + '.npy'
        question_str = annotations[key]['question']
        question_tokens = text_processing.tokenize(question_str)

        iminfo = dict(image_name=image_name,
                    image_id=image_id,
                    question_id=question_id,
                    feature_path=feature_path,
                    question_str=question_str,
                    question_tokens=question_tokens)

        if load_answer:
            answer = annotations[key]['answer']
            answers = [answer] * 10 # make it like VQA dataset
            all_answers, valid_answers = extract_answers(answers,
                                                            valid_answer_set)
            if len(valid_answers) == 0:
                valid_answers = ['<unk>']
                unk_ans_count += 1

            iminfo['all_answers'] = all_answers
            iminfo['valid_answers'] = valid_answers
            has_answer = True
        if load_gt_layout:
            has_gt_layout = True

        imdb[n_q+1] = iminfo
    print('total %d out of %d answers are <unk>' % (unk_ans_count,
                                                    len(annotations)))
    header = create_header("gqa", has_answer=has_answer,
                        has_gt_layout=has_gt_layout)
    imdb[0] = header
    return imdb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="data directory")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="imdb output directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir

    vocab_answer_file = os.path.join(out_dir, 'answers_gqa_9.txt')
    answer_dict = text_processing.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list)

    # imdb_train_gqa = build_imdb('train_balanced', valid_answer_set)
    imdb_val_gqa = build_imdb('val', valid_answer_set)
    # imdb_test_gqa = build_imdb('test', valid_answer_set)

    imdb_dir = os.path.join(out_dir, 'imdb')
    os.makedirs(imdb_dir, exist_ok=True)
    
    
    # np.save(os.path.join(imdb_dir, 'imdb_train_balanced_gqa_9.npy'),
            # np.array(imdb_train_gqa))
    np.save(os.path.join(imdb_dir, 'imdb_val_all_questions_9.npy'),
            np.array(imdb_val_gqa))
    # np.save(os.path.join(imdb_dir, 'imdb_submission_all_questions_gqa.npy'),
    #         np.array(imdb_test_gqa))