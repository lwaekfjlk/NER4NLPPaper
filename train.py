from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
import argparse
from dataset import ScirexDataset, ConLLDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os

def load_dataset(args, tokenizer):
    '''
    loading datasets, return a dictionary of dataloaders
    '''
    loader_dict = {}

    if args.train:
        if args.dataset == 'scirex':
            train_dataset = ScirexDataset(args.train_file, tokenizer)
            dev_dataset = ScirexDataset(args.dev_file, tokenizer)
        elif args.dataset == 'conll':
            train_dataset = ConLLDataset(args.train_file, tokenizer)
            dev_dataset = ConLLDataset(args.dev_file, tokenizer)
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
        elif args.dataset == 'conll':
            test_dataset = ConLLDataset(args.test_file, tokenizer)
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
    for data in tqdm(dev_dataloader):
        input_ids = data['input_ids'].to(args.device)
        labels = data['labels'].to(args.device)
        mask_ids = data[['attention_mask']].to(args.device)
        outputs = model(input_ids, labels=labels, attention_mask=mask_ids)
        loss = outputs[0]
        print(f'Validation loss: {loss}')


def train(args, model, tokenizer):
    print('=====begin loading and tokenizing dataset====')
    loaders = load_dataset(args, tokenizer)
    print('=====end loading and tokenizing dataset====')
    train_dataloader = loaders['train']
    dev_dataloader = loaders['dev']
    model.train()
    optimizer = attach_optimizer(args, model)

    for epoch in tqdm(range(args.num_epochs)):
        for data in tqdm(train_dataloader):
            input_ids = data['input_ids'].to(args.device)
            labels = data['labels'].to(args.device)
            mask_ids = data['attention_mask'].to(args.device)
            outputs = model(input_ids, labels=labels, attention_mask=mask_ids)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch} loss: {loss}')


def inference(args, model, tokenizer):
    loaders = load_dataset(args, tokenizer)
    test_dataloader = loaders['test']


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
    parser.add_argument('--train_file', type=str, default='./data/train.jsonl', help='path to train file, jsonl for scirex, conll for conll')
    parser.add_argument('--dev_file', type=str, default='./data/dev.jsonl', help='path to dev file')
    parser.add_argument('--test_file', type=str, default='./data/test.conll', help='path to test file')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='../output')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='scirex')
    parser.add_argument('--label_num', type=int, default=9, help='number of labels, 15 for conll, 9 for scirex')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    args.local_rank = int(os.environ['LOCAL_RANK'])


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=args.label_num, ignore_mismatched_sizes=True)
    device = torch.device(args.local_rank) if args.local_rank != -1 else torch.device('cuda')
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        distributed_setup(args, model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    

    if args.train:
        print('hello')
        train(args, model, tokenizer)


    # nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = 'My name is Clara and I live in Berkeley, California.'

    # ner_results = nlp(example)
