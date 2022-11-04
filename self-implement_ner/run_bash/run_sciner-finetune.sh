export CUDA_VISIBLE_DEVICES=1
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29519 ../main.py \
--train \
--use_wandb \
--model_name allenai/scibert_scivocab_uncased \
--dataset sciner \
--train_file ../data/sciner_dataset/train.conll \
--dev_file ../data/sciner_dataset/validation.conll \
--test_file ../data/sciner_dataset/validation.conll \
--checkpoint_save_dir ../checkpoints/ \
--task sciner-finetune \
--train_batch_size 8 \
--gradient_accumulation_step 4 \
--model_chosen_metric f1 \
--dev_batch_size 8 \
--max_length 512 \
--num_epochs 30 \
--model_type bertbilstmcrf \
--learning_rate 5e-5 \
--evaluation_steps 50
