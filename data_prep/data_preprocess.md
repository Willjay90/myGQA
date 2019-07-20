<!-- 
 pip install numpy==1.16.1
 export PYTHONPATH=. 
-->

# question
python data_prep/extract_vocabulary.py \
--input_files data/questions1.2/val_all_questions.json \
 data/questions1.2/test_all_questions.json \
 data/questions1.2/challenge_all_questions.json \
 data/questions1.2/testdev_all_questions.json \
 data/questions1.2/submission_all_questions.json \
 data/questions1.2/train_balanced_questions.json
--out_dir data/

<!-- 
min question len= 3
max question len= 29 
-->


# answer
python GQA/data_prep/process_answer.py \
--annotation_file GQA/data/questions1.2/val_all_questions.json \
 GQA/data/questions1.2/val_all_questions.json \
 GQA/data/questions1.2/train_balanced_questions.json \
--out_dir data/ --min_freq 9

# embedding
python GQA/data_prep/extract_word_glove_embedding.py  \
--vocabulary_file data/vocabulary_gqa.txt  \
--glove_file data/glove/glove.6B.300d.txt \
--out_dir data/

# imdb
python GQA/data_prep/build_gqa_imdb.py --data_dir GQA/data/questions1.2/ --out_dir data/

<!-- answer > 3
building imdb train
total 2451 out of 14305356 answers are <unk>

building imdb val
total 298 out of 2011853 answers are <unk>

building imdb test
total 0 out of 1340048 answers are <unk>

-->

<!-- answer > 0: 1853

building imdb train_balanced
total 882 out of 943000 answers are <unk>

building imdb val
total 175 out of 2011853 answers are <unk>

building imdb val_balanced
total 99 out of 132062 answers are <unk> 

-->


<!-- answer > 9: 1555
building imdb train_balanced
total 1932 out of 943000 answers are <unk>

building imdb val
total 371 out of 2011853 answers are <unk>

building imdb val_balanced
total 239 out of 132062 answers are <unk>
-->
