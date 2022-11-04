export CUDA_VISIBLE_DEVICES=1
export NGPU=1
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port 29519 ../main.py \
--inference \
--task sciner-finetune \
--model_name allenai/scibert_scivocab_uncased \
--dataset sciner \
--train_file ../data/sciner_dataset/train.conll \
--dev_file ../data/sciner_dataset/validation.conll \
--test_file ../data/sciner_dataset/validation.conll \
--checkpoint_save_dir ../checkpoints/ \
--inference_file ../data/anlp_test/anlp-sciner-test-sentences.txt \
--output_file ../data/anlp_test/anlp-sciner-test-sentences.conll \
--label_num 15

python ../utils/sentence_to_paragraph.py