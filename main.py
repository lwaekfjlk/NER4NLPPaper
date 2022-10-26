import os
import torch
import argparse
import time
import csv
import shutil
import evaluate
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline
from dataset import ScirexDataset, SciNERDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP



def load_dataset(args, tokenizer):
    '''
    loading datasets, return a dictionary of dataloaders
    '''
    loader_dict = {}

    if args.train:
        if args.dataset == 'scirex':
            train_dataset = ScirexDataset(args.train_file, tokenizer)
            dev_dataset = ScirexDataset(args.dev_file, tokenizer)
        elif args.dataset == 'sciner':
            train_dataset = SciNERDataset(args.train_file, tokenizer)
            dev_dataset = SciNERDataset(args.dev_file, tokenizer)
        else:
            raise ValueError('Invalid dataset')
        if torch.cuda.device_count() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
            dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=lambda x: train_dataset.collate_fn(x, args.max_length))
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, sampler=dev_sampler, collate_fn=lambda x: dev_dataset.collate_fn(x, args.max_length))
        else:
            train_dataloader = DataLoader(train_dataset,  batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: train_dataset.collate_fn(x, args.max_length))
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: dev_dataset.collate_fn(x, args.max_length))
        loader_dict['train'] = train_dataloader
        loader_dict['dev'] = dev_dataloader
        loader_dict['train'] = train_dataloader
        loader_dict['dev'] = dev_dataloader

    if args.inference:
        if args.dataset == 'scirex':
            test_dataset = ScirexDataset(args.test_file, tokenizer)
        elif args.dataset == 'sciner':
            test_dataset = SciNERDataset(args.test_file, tokenizer)
        else:
            raise ValueError('Invalid dataset')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
        loader_dict['test'] = test_dataloader
    
    return loader_dict


def attach_optimizer(args, model):
    '''
    attach optimizer to the model
    '''
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError('Invalid optimizer')

    return optimizer


def validate(args, dev_dataloader, model):
    correct_ones = 0
    all_ones = 0
    eval_losses = []
    gth_labels = []
    pred_labels = []
    for data in dev_dataloader:
        input_ids = data['input_ids'].to(args.device)
        labels = data['labels'].to(args.device)
        mask_ids = data['attention_mask'].to(args.device)
        outputs = model(input_ids, labels=labels, attention_mask=mask_ids)
        eval_loss = outputs['loss']
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=-1)
        pred_labels += predictions.view(-1).tolist()
        gth_labels += labels.view(-1).tolist()
        eval_losses.append(eval_loss.item()) 
    f1_metric = evaluate.load("f1")
    results = f1_metric.compute(
        predictions=pred_labels, 
        references=gth_labels, 
        labels=[i for i in range(args.label_num)],
        average='weighted',
    )
    f1 = results['f1']
    print(f'validation f1 : {f1}')
    eval_loss = sum(eval_losses) / len(eval_losses)
    print(f'validation loss : {eval_loss}')
    return f1, eval_loss


def train(args, model, tokenizer):
    best_checkpoint_name = None
    best_eval_f1 = -1
    global_step = 0
    print('=====begin loading dataset====')
    loaders = load_dataset(args, tokenizer)
    print('=====end loading dataset====')
    train_dataloader = loaders['train']
    dev_dataloader = loaders['dev']
    model.train()
    optimizer = attach_optimizer(args, model)

    train_losses = []
    for epoch in range(args.num_epochs):
        for data in tqdm(train_dataloader):
            input_ids = data['input_ids'].to(args.device)
            labels = data['labels'].to(args.device)
            mask_ids = data['attention_mask'].to(args.device)
            outputs = model(input_ids, labels=labels, attention_mask=mask_ids, return_dict=True)
            loss = outputs['loss']
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % args.evaluation_steps == 0:
                eval_f1, eval_loss = validate(args, dev_dataloader, model)
                if eval_f1 > best_eval_f1:
                    if best_checkpoint_name is not None:
                        os.remove(best_checkpoint_name)
                        best_checkpoint_name = args.checkpoint_save_dir + 'best_model4{}_f1_{}_{}.ckpt'.format(args.task, round(eval_f1*100,3), args.timestamp)
                    else:
                        best_checkpoint_name = args.checkpoint_save_dir + 'best_model4{}_f1_{}_{}.ckpt'.format(args.task, round(eval_f1*100,3), args.timestamp)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = best_checkpoint_name
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_eval_f1 = eval_f1
        epoch_loss = sum(train_losses) / len(train_losses)
        print(f'Epoch {epoch} loss: {epoch_loss}')

    src_file = best_checkpoint_name
    tgt_file = args.checkpoint_save_dir + 'best_model4{}.ckpt'.format(args.task)
    shutil.copy(src_file, tgt_file)
    return


def test(args, model, tokenizer):
    raise NotImplementedError


def sciner_inference(args, model, tokenizer):
    entities = [
        'O',
        'B-MethodName', 'I-MethodName', 'B-HyperparameterName', 'I-HyperparameterName',
        'B-HyperparameterValue', 'I-HyperparameterValue', 'B-MetricName', 'I-MetricName',
        'B-MetricValue', 'I-MetricValue', 'B-TaskName', 'I-TaskName', 'B-DatasetName', 'I-DatasetName'
    ]

    model.load_state_dict(torch.load(args.checkpoint_save_dir + 'best_model4{}.ckpt'.format(args.task)))
    id2entity = {i: e for i, e in enumerate(entities)}
    label2id = model.config.label2id

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=0)
    with open(args.output_file, 'w', newline='') as output_f, open(args.inference_file, 'r') as input_f:
        sents = input_f.readlines()
        for sent in sents:
            target_words = sent.strip().split(' ')
            ner_res = ner_pipeline(sent)
            words = []
            entities = []
            ner_index = 0
            target_index = 0
            while ner_index < len(ner_res):
                sub_word = ner_res[ner_index]['word']
                sub_word = sub_word.replace('##', '')
                entity = ner_res[ner_index]['entity']
                entity = id2entity[label2id[entity]]
                words.append(sub_word)
                entities.append(entity)
                ner_index += 1
                target_index += 1
                while words[-1] != target_words[target_index-1]:
                    sub_word = ner_res[ner_index]['word']
                    words[-1] += sub_word.replace('##', '')
                    ner_index += 1
                output_f.write(words[-1]+'\t'+entities[-1]+'\n')
            output_f.write('\n')
    return


def distributed_setup(args, model):
    '''
    setup distributed training
    '''
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.device = torch.device('cuda', args.local_rank)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dslim/bert-base-NER', help='model name or path')
    parser.add_argument('--train_file', type=str, default='./data/sciner_dataset/train.conll', help='path to train file, jsonl for scirex, conll for sciner')
    parser.add_argument('--dev_file', type=str, default='./data/sciner_dataset/validation.conll', help='path to dev file')
    parser.add_argument('--test_file', type=str, default='./data/sciner_dataset/validation.conll', help='path to test file')
    parser.add_argument('--inference_file', type=str, default='./data/anlp_test/anlp-sciner-test.txt', help='final ANLP submission file')
    parser.add_argument('--output_file', type=str, default='./data/anlp_test/anlp_haofeiy.conll')
    parser.add_argument('--task', type=str, default='sciner-finetune', choices=['sciner-finetune', 'scirex-finetune'])
    parser.add_argument('--load_from_checkpoint', type=str, default=None, help='contine finetuning based on one checkpoint')
    parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='sciner')
    parser.add_argument('--label_num', type=int, default=15, help='number of labels, 15 for sciner, 9 for scirex')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation_steps', type=int, default=50)
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time()*1000))/1000))

    if args.use_wandb:
        import wandb
        # need to change to your own API when using
        os.environ['EXP_NUM'] = 'SciNER'
        os.environ['WANDB_NAME'] = time.strftime(
            '%Y-%m-%d %H:%M:%S', 
            time.localtime(int(round(time.time()*1000))/1000)
        )
        os.environ['WANDB_API_KEY'] = '972035264241fb0f6cc3cab51a5d82f47ca713db'
        os.environ['WANDB_DIR'] = './SciNER_tmp'
        wandb.init(project="SciNER")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.label_num)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)
    device = torch.device(args.local_rank) if args.local_rank != -1 else torch.device('cuda')
    if args.load_from_checkpoint:
        model_dict = torch.load(args.load_from_checkpoint)
        filtered_model_dict = {k: v for k, v in model_dict.items() if 'classifier' not in k}
        model_dict.update(filtered_model_dict)
        model.load_state_dict(filtered_model_dict, strict=False)
    
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        distributed_setup(args, model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.train:
        train(args, model, tokenizer)
    elif args.inference:
        conll_result = sciner_inference(args, model, tokenizer)
