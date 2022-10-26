export CUDA_VISIBLE_DEVICES=0
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29519 train.py --train --use_wandb
