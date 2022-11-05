export CUDA_VISIBLE_DEVICES=1
python ../main.py \
--inference \
--task sciner-finetune \
--model_type bertcrf \
--model_name allenai/scibert_scivocab_uncased \
--dataset sciner \
--train_file ../data/sciner_dataset/train.conll \
--dev_file ../data/sciner_dataset/validation.conll \
--test_file ../data/sciner_dataset/validation.conll \
--ckpt_save_dir ../checkpoints/ \
--inference_file ../data/anlp_valid/anlp-sciner-valid-sentences.txt \
--output_file ../data/anlp_valid/anlp-sciner-valid-sentences.conll