export CUDA_VISIBLE_DEVICES=0
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29519 main.py \
--train \
--use_wandb \
--dataset sciner \
--train_file ./data/sciner_dataset/train.conll \
--dev_file ./data/sciner_dataset/validation.conll \
--test_file ./data/sciner_dataset/validation.conll \
--checkpoint_save_dir ./checkpoints/ \
--task sciner-finetune \
--train_batch_size 8 \
--gradient_accumulation_step 1 \
--dev_batch_size 8 \
--max_length 512 \
--num_epochs 30 \
--learning_rate 1e-5 \
--label_num 17 \
--evaluation_steps 50 \
--load_from_checkpoint ./checkpoints/best_model4scirex-finetune.ckpt \
