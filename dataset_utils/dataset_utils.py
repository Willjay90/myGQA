from dataset_utils.gqa_dataset import GQADataSet

def prepare_data_set(data_dir, root_dir, **data_param):
    gqa_dataset = GQADataSet(input_file=data_dir, root_dir=root_dir, **data_param)
    vocab_question_file = "vocabulary_gqa.txt"
    vocab_answer_file = "answers_gqa.txt"

    return gqa_dataset

def prepare_train_data_set(root_dir, **data_param):
    return prepare_data_set(root_dir + "/questions1.2/train_balanced_questions.json", root_dir, **data_param)


def prepare_eval_data_set(root_dir, **data_param):
    return prepare_data_set(root_dir + "/questions1.2/val_balanced_questions.json", root_dir, **data_param)


def prepare_test_data_set(root_dir, **data_param):
    return prepare_data_set(root_dir + "/questions1.2/submission_all_questions.json", root_dir, **data_param)