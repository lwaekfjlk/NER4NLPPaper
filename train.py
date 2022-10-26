import os
import torch
import argparse
import time
import wandb
import evaluate
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from dataset import ScirexDataset, ConLLDataset
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
        for data in train_dataloader:
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
                    if os.path.exists('./checkpoints/best_NER_model_f1_{}.ckpt'.format(round(best_eval_f1*100, 3))):
                        os.remove('./checkpoints/best_NER_model_f1_{}.ckpt'.format(round(best_eval_f1*100, 3)))
                    torch.save(model.state_dict(), './checkpoints/best_NER_model_f1_{}.ckpt'.format(round(eval_f1*100,3)))
                    best_eval_f1 = eval_f1
        epoch_loss = sum(train_losses) / len(train_losses)
        print(f'Epoch {epoch} loss: {epoch_loss}')


def test(args, model, tokenizer):
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
    parser.add_argument('--train_file', type=str, default='./data/conll_dataset/train.conll', help='path to train file, jsonl for scirex, conll for conll')
    parser.add_argument('--dev_file', type=str, default='./data/conll_dataset/validation.conll', help='path to dev file')
    parser.add_argument('--test_file', type=str, default='./data/conll_dataset/validation.conll', help='path to test file')
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
    parser.add_argument('--dataset', type=str, default='conll')
    parser.add_argument('--label_num', type=int, default=15, help='number of labels, 15 for conll, 9 for scirex')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation_steps', type=int, default=50)
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.use_wandb:
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
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=args.label_num, ignore_mismatched_sizes=True)
    device = torch.device(args.local_rank) if args.local_rank != -1 else torch.device('cuda')
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        distributed_setup(args, model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    

    if args.train:
        train(args, model, tokenizer)


    # nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = 'My name is Clara and I live in Berkeley, California.'

    # ner_results = nlp(example)
