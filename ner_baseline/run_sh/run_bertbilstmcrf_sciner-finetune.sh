export CUDA_VISIBLE_DEVICES=2
python ../main.py \
--train \
--use_fp16 \
--use_wandb \
--model_type bertbilstmcrf \
--model_name allenai/scibert_scivocab_uncased \
--dataset sciner \
--train_file ../data/sciner_dataset/train.conll \
--dev_file ../data/sciner_dataset/validation.conll \
--test_file ../data/sciner_dataset/validation.conll \
--ckpt_save_dir ../checkpoints/ \
--task sciner-finetune \
--train_batch_size 8 \
--gradient_accumulation_step 1 \
--model_chosen_metric f1 \
--dev_batch_size 8 \
--max_length 512 \
--num_epochs 50 \
--learning_rate 3e-5 \
--crf_learning_rate 1e-2 \
--evaluation_steps 50
