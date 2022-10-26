export CUDA_VISIBLE_DEVICES=2
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29519 main.py \
--train \
--use_wandb \
--dataset scirex \
--train_file ./data/scirex_dataset/train.jsonl \
--dev_file ./data/scirex_dataset/dev.jsonl \
--test_file ./data/scirex_dataset/test.jsonl \
--checkpoint_save_dir ./checkpoints/ \
--task scirex-finetune \
--batch_size 8 \
--max_length 512 \
--num_epochs 10 \
--learning_rate 1e-5 \
--label_num 9 \
--evaluation_steps 500 \
