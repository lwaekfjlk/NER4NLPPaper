import os
import torch
import argparse
import time
import csv
import shutil
import evaluate
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline
from transformers import get_cosine_schedule_with_warmup
from utils.dataset import ScirexDataset, SciNERDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchcrf import CRF
from model import BertCRF, BertBiLSTMCRF, BertSoftmax

import warnings
warnings.filterwarnings('ignore')


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
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=train_sampler, collate_fn=lambda x: train_dataset.collate_fn(x, args))
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, sampler=dev_sampler, collate_fn=lambda x: dev_dataset.collate_fn(x, args))
        else:
            train_dataloader = DataLoader(train_dataset,  batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda x: train_dataset.collate_fn(x, args))
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=True, collate_fn=lambda x: dev_dataset.collate_fn(x, args))
        loader_dict['train'] = train_dataloader
        loader_dict['dev'] = dev_dataloader

    if args.inference:
        if args.dataset == 'scirex':
            test_dataset = ScirexDataset(args.test_file, tokenizer)
        elif args.dataset == 'sciner':
            test_dataset = SciNERDataset(args.test_file, tokenizer)
        else:
            raise ValueError('Invalid dataset')
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
        loader_dict['test'] = test_dataloader
    
    return loader_dict


def attach_optimizer(args, model):
    '''
    attach optimizer to the model
    '''
    if args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError('Invalid optimizer')

    return optimizer


def attach_scheduler(args, optimizer, total_training_steps):
    '''
    attach lr scheduler to the model
    '''
    if args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps,
        )
        return scheduler
    else:
        raise ValueError('Invalid scheduler type')



def validate(args, dev_dataloader, model):
    model.eval()

    eval_losses = []
    refs = []
    preds = []
    with torch.no_grad():
        for batch in dev_dataloader:
            outputs = model(
                input_ids=batch['input_ids'],
                token_starts=batch['token_starts'],
                labels=batch['labels'], 
                attention_mask=batch['attention_mask'],
            )
            eval_loss = outputs[0]
            logits = outputs[1]
            pred = torch.argmax(logits, dim=-1)
            ref = batch['labels']
            eval_losses.append(eval_loss.item()) 
            preds.append(pred.view(-1).tolist())
            refs.append(ref.view(-1).tolist())
            

    metric = evaluate.load("seqeval")
    predictions = [
        [args.id2entity[p] for (p, l) in zip(pred, ref) if l != -100]
        for pred, ref in zip(preds, refs)
    ]
    references = [
        [args.id2entity[l] for (p, l) in zip(pred, ref) if l != -100]
        for pred, ref in zip(preds, refs)
    ]
    f1 = metric.compute(predictions=predictions, references=references)['overall_f1']

    eval_loss = sum(eval_losses) / len(eval_losses)
    
    print(f'validation f1 : {f1}')
    print(f'validation loss : {eval_loss}')
    return f1, eval_loss


def train(args, model, tokenizer):
    best_checkpoint_name = None
    best_eval_f1 = -float('inf')
    best_eval_loss = float('inf')
    step = 0
    iteration = 0
    print('=====begin loading dataset====')
    loaders = load_dataset(args, tokenizer)
    print('=====end loading dataset====')
    train_dataloader = loaders['train']
    dev_dataloader = loaders['dev']
    total_training_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_step
    optimizer = attach_optimizer(args, model)
    scheduler = attach_scheduler(args, optimizer, total_training_steps)
    model.train()
    

    train_losses = []
    for epoch in range(args.num_epochs):
        for batch in tqdm(train_dataloader):
            model.train()
            outputs = model(
                input_ids=batch['input_ids'],
                token_starts=batch['token_starts'],
                labels=batch['labels'], 
                attention_mask=batch['attention_mask'],
            )
            loss = outputs[0]
            loss.backward()
            train_losses.append(loss.item())
            if args.use_wandb:
                wandb.log({'train loss': loss.item(), 'iteration': iteration})
            iteration += 1
            if iteration % args.gradient_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if args.use_wandb:
                    wandb.log({'learning rate': scheduler.get_last_lr()[0], 'step': step})

                if step % args.evaluation_steps == 0:
                    eval_f1, eval_loss = validate(args, dev_dataloader, model)
                    if args.use_wandb:
                        wandb.log({'eval_f1': eval_f1, 'step': step})
                        wandb.log({'eval_loss': eval_loss, 'step': step})
                    if args.model_chosen_metric == 'f1':
                        if eval_f1 > best_eval_f1:
                            if best_checkpoint_name is not None:
                                os.remove(best_checkpoint_name)
                                best_checkpoint_name = args.checkpoint_save_dir + 'best_{}4{}_f1_{}_{}.ckpt'.format(args.model_name.split('/')[-1], args.task, round(eval_f1*100,3), args.timestamp)
                            else:
                                best_checkpoint_name = args.checkpoint_save_dir + 'best_{}4{}_f1_{}_{}.ckpt'.format(args.model_name.split('/')[-1], args.task, round(eval_f1*100,3), args.timestamp)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = best_checkpoint_name
                            torch.save(model_to_save.state_dict(), output_model_file)
                            best_eval_f1 = eval_f1
                    elif args.model_chosen_metric == 'loss':
                        if eval_loss < best_eval_loss:
                            if best_checkpoint_name is not None:
                                os.remove(best_checkpoint_name)
                                best_checkpoint_name = args.checkpoint_save_dir + 'best_{}4{}_loss_{}_{}.ckpt'.format(args.model_name.split('/')[-1], args.task, round(eval_loss,3), args.timestamp)
                            else:
                                best_checkpoint_name = args.checkpoint_save_dir + 'best_{}4{}_loss_{}_{}.ckpt'.format(args.model_name.split('/')[-1], args.task, round(eval_loss,3), args.timestamp)
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = best_checkpoint_name
                            torch.save(model_to_save.state_dict(), output_model_file)
                            best_eval_loss = eval_loss
                    else:
                        raise NotImplementedError
        epoch_loss = sum(train_losses) / len(train_losses)
        print(f'Epoch {epoch} loss: {epoch_loss}')

    src_file = best_checkpoint_name
    tgt_file = args.checkpoint_save_dir + 'best_{}4{}.ckpt'.format(args.model_name.split('/')[-1], args.task)
    shutil.copy(src_file, tgt_file)
    return


def ner_inference(args, sent, model, tokenizer):
    tokenized_sent = tokenizer.tokenize(sent)
    input_ids = tokenizer.encode(sent)
    token_starts = [0] + [1 - int(token.startswith('##')) for token in tokenized_sent] + [0]
    input_ids = torch.LongTensor([input_ids])
    input_ids = input_ids[:, :args.max_length].to(args.device)
    token_starts = torch.ByteTensor([token_starts])
    token_starts = token_starts[:, :args.max_length].to(args.device)
    outputs = model(
        input_ids=input_ids,
        token_starts=token_starts,
    )
    logits = outputs[-1]
    entities = []
    words = []
    preds = torch.argmax(logits, dim=-1)[0].tolist()

    token_starts = [1 - int(token.startswith('##')) for token in tokenized_sent]
    for token_start, token in zip(token_starts, tokenized_sent):
        if token_start == 0:
            entities.append('O')
            words.append(token.replace('##', ''))
        else:
            entities.append(args.id2entity[preds.pop(0)])
            words.append(token)

    assert len(entities) == len(words)
    return words, entities



def sciner_inference(args, model, tokenizer):
    def unk_wrapper(word):
        return tokenizer.decode(tokenizer.encode(word), skip_special_tokens=True)

    model.load_state_dict(torch.load(args.checkpoint_save_dir + 'best_{}4{}.ckpt'.format(args.model_name.split('/')[-1], args.task)))
    with open(args.output_file, 'w', newline='') as output_f, open(args.inference_file, 'r') as input_f:
        sents = input_f.readlines()
        for sent in tqdm(sents):
            words = sent.strip().split(' ')
            src_words, src_entities = ner_inference(args, sent, model, tokenizer)
            tgt_words = []
            tgt_entities = []
            src_index = 0
            tgt_index = 0
            while src_index < len(src_words):
                output_word = words[tgt_index]
                output_entity = src_entities[src_index]
                tgt_words.append(src_words[src_index])
                tgt_entities.append(src_entities[src_index])
                matcher = unk_wrapper(words[tgt_index])
                matchee = unk_wrapper(tgt_words[-1])
                src_index += 1
                tgt_index += 1
                while matcher != matchee:
                    tgt_words[-1] += src_words[src_index]
                    src_index += 1
                    matchee = unk_wrapper(tgt_words[-1])
                output_f.write(output_word + '\t' + output_entity + '\n')
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
    parser.add_argument('--model_name', type=str, default='allenai/scibert_scivocab_uncased', help='model name or path')
    parser.add_argument('--train_file', type=str, default='./data/sciner_dataset/train.conll', help='path to train file, jsonl for scirex, conll for sciner')
    parser.add_argument('--dev_file', type=str, default='./data/sciner_dataset/validation.conll', help='path to dev file')
    parser.add_argument('--test_file', type=str, default='./data/sciner_dataset/validation.conll', help='path to test file')
    parser.add_argument('--inference_file', type=str, default='./data/anlp_test/anlp-sciner-test.txt', help='final ANLP submission file')
    parser.add_argument('--output_file', type=str, default='./data/anlp_test/anlp_haofeiy_sciner.conll')
    parser.add_argument('--task', type=str, default='sciner-finetune', choices=['sciner-finetune', 'scirex-finetune'])
    parser.add_argument('--load_from_checkpoint', type=str, default=None, help='contine finetuning based on one checkpoint')
    parser.add_argument('--model_chosen_metric', type=str, default='f1', help='choose dev checkpoint based on this metric')
    parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints/')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_step', type=int, default=4)
    parser.add_argument('--dev_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--optimizer_type', type=str, default='adamw')
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='sciner')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation_steps', type=int, default=50)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--model_type', type=str, default='bertsoftmax', choices=['bertsoftmax', 'bertbilstmcrf', 'bertcrf'])
    parser.add_argument('-n', '--id2entity', nargs='+', default=[
        'O',
        'B-MethodName', 'I-MethodName', 'B-HyperparameterName', 'I-HyperparameterName',
        'B-HyperparameterValue', 'I-HyperparameterValue', 'B-MetricName', 'I-MetricName',
        'B-MetricValue', 'I-MetricValue', 'B-TaskName', 'I-TaskName', 'B-DatasetName', 'I-DatasetName',
    ])

    args = parser.parse_args()
    if torch.cuda.device_count() > 1:
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
        os.environ['WANDB_DIR'] = '../SciNER_tmp'
        wandb.init(project="SciNER")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=len(args.id2entity))
    device = torch.device(args.local_rank) if args.local_rank != -1 else torch.device('cuda')
    if args.model_type == 'bertsoftmax':
            model = BertSoftmax.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)
    elif args.model_type == 'bertbilstmcrf':
            model = BertBiLSTMCRF.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)
    elif args.model_type == 'bertcrf':
            model = BertCRF.from_pretrained(args.model_name, config=config, ignore_mismatched_sizes=True)
    model.to(device)
    
    if args.load_from_checkpoint:
        model_dict = torch.load(args.load_from_checkpoint)
        filtered_model_dict = {k: v for k, v in model_dict.items() if 'classifier' not in k}
        model_dict.update(filtered_model_dict)
        model.load_state_dict(filtered_model_dict, strict=False)
    
    
    if torch.cuda.device_count() > 1:
        distributed_setup(args, model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.train:
        train(args, model, tokenizer)
    elif args.inference:
        conll_result = sciner_inference(args, model, tokenizer)
