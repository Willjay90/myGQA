## GQA Chanllenge
This work is motivated by [BERT](https://arxiv.org/abs/1810.04805) and [Transformer](https://arxiv.org/abs/1706.03762). I modified the transformer architecture, adding a image feature attention block right after the MultiHead Attention block in both encoder and decoder.

## Data Preprocessing 

1) Prepare Question Vocabulary

```
python data_prep/extract_vocabulary.py --type question --input_files data/questions1.2/train_balanced_questions.json data/questions1.2/val_balanced_questions.json data/questions1.2/submission_all_questions.json
```

2) Prepare Answer Vocabulary

```
python data_prep/extract_vocabulary.py --type answer --input_files data/questions1.2/train_balanced_questions.json data/questions1.2/val_balanced_questions.json
```

## Training

All training parameters is located in `config/config.py`

```
python train.py
```


## Generate Test Result

```
python test.py --config results/1234/config.yaml --out_prefix gqa_transformer --model_path results/1234/best_model.pth
```

## Acknowledgments

My implementation utilizes code from the following:

* [OpenNMT](https://github.com/OpenNMT/OpenNMT)
* [Pythia](https://github.com/facebookresearch/pythia)
* [Harvardnlp](http://nlp.seas.harvard.edu/2018/04/03/attention.html)