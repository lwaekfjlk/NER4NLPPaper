import os
import torch
import argparse
import time
import csv
import shutil
import evaluate
import logging
import wandb
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoConfig
from transformers import AdamW, get_cosine_schedule_with_warmup
from dataset import SciNERDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from model import BertCRF, BertBiLSTMCRF, Bert

import warnings
warnings.filterwarnings('ignore')


def set_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_wandb(args):
    if args.use_wandb:
        # need to change to your own API when using
        os.environ['EXP_NUM'] = 'SciNER'
        os.environ['WANDB_NAME'] = args.timestamp
        os.environ['WANDB_API_KEY'] = '972035264241fb0f6cc3cab51a5d82f47ca713db'
        os.environ['WANDB_DIR'] = '../SciNER_tmp'
        wandb.init(project="SciNER")
    return


def attach_dataloader(args, tokenizer):
    loader_dict = {}
    if args.train:
        train_dataset = SciNERDataset(args, tokenizer, 'train')
        dev_dataset = SciNERDataset(args, tokenizer, 'dev')
        train_dataloader = DataLoader(
            train_dataset,  
            batch_size=args.train_batch_size, 
            shuffle=True, 
            collate_fn=lambda x: train_dataset.collate_fn(x, args)
        )
        dev_dataloader = DataLoader(
            dev_dataset, 
            batch_size=args.dev_batch_size, 
            shuffle=True, 
            collate_fn=lambda x: dev_dataset.collate_fn(x, args)
        )
        loader_dict['train'] = train_dataloader
        loader_dict['dev'] = dev_dataloader

    if args.inference:
        test_dataset = SciNERDataset(args, tokenizer, 'test')
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size, 
            shuffle=False, 
            collate_fn=test_dataset.collate_fn
        )
        loader_dict['test'] = test_dataloader
    
    return loader_dict


def attach_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.model_name)


def attach_model(args):
    config = AutoConfig.from_pretrained(args.model_name, num_labels=len(args.id2entity))
    if args.model_type == 'bert':
        model = Bert.from_pretrained(
            args.model_name, 
            config=config, 
            ignore_mismatched_sizes=True
        )
    elif args.model_type == 'bertcrf':
        model = BertCRF.from_pretrained(
            args.model_name, 
            config=config, 
            ignore_mismatched_sizes=True
        )
    elif args.model_type == 'bertbilstmcrf':
        model = BertBiLSTMCRF.from_pretrained(
            args.model_name, 
            config=config, 
            ignore_mismatched_sizes=True
        )
    device = torch.device('cuda')
    model.to(device)

    if args.load_from_ckpt:
        model_dict = torch.load(args.load_from_ckpt)
        bert_model_dict = {k: v for k, v in model_dict.items() if 'classifier' not in k}
        model_dict.update(bert_model_dict)
        model.load_state_dict(bert_model_dict, strict=False)
    return model


def attach_optimizer(args, model):
    if args.optimizer_type == 'adamw':
        bert_optimizer = list(model.bert.named_parameters())
        classifier_optimizer = list(model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': args.learning_rate * 5, 'weight_decay': args.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': args.learning_rate * 5, 'weight_decay': 0.0},
        ]
        if args.model_type == 'bertcrf':
            optimizer_grouped_parameters += [
                {'params': model.crf.parameters(), 'lr': args.crf_learning_rate},
            ]
        elif args.model_type == 'bertbilstmcrf':
            lstm_optimizer = list(model.bilstm.named_parameters())
            optimizer_grouped_parameters += [
                {'params': model.crf.parameters(), 'lr': args.crf_learning_rate},
                {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
                'lr': args.learning_rate * 5, 'weight_decay': args.weight_decay},
                {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
                'lr': args.learning_rate * 5, 'weight_decay': 0.0},
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        raise ValueError('Invalid optimizer type')
    return optimizer


def attach_scheduler(args, optimizer, train_dataloader):
    train_steps_per_epoch = len(train_dataloader)
    total_training_steps = args.num_epochs * train_steps_per_epoch // args.gradient_accumulation_step
    total_warmup_steps = (args.num_epochs // 5) * train_steps_per_epoch // args.gradient_accumulation_step
    if args.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_warmup_steps,
            num_training_steps=total_training_steps,
        )
        return scheduler
    else:
        raise ValueError('Invalid scheduler type')


def save_model(best_ckpt_name, metric, best_metric):
    eps = 1e-5
    if (args.model_chosen_metric == 'f1' and metric['f1'] > best_metric['f1'] + eps) or \
       (args.model_chosen_metric == 'loss' and metric['loss'] < best_metric['loss'] - eps):
        if best_ckpt_name is not None:
            os.remove(os.path.join(args.ckpt_save_dir,best_ckpt_name))
        best_ckpt_name = 'best_{}4{}_{}_{}_{}.ckpt'.format(
            args.model_type, 
            args.task, 
            args.model_chosen_metric, 
            round(metric[args.model_chosen_metric],3), 
            args.timestamp
        )
        output_model_file = os.path.join(args.ckpt_save_dir, best_ckpt_name)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), output_model_file)
        best_metric[args.model_chosen_metric] = metric[args.model_chosen_metric]
    return best_ckpt_name, best_metric


def save_final_model(best_ckpt_name):
    src_file = os.path.join(args.ckpt_save_dir, best_ckpt_name)
    tgt_file = os.path.join(args.ckpt_save_dir, 'best_{}4{}.ckpt'.format(args.model_type, args.task))
    shutil.copy(src_file, tgt_file)
    return


def validate(args, dev_dataloader, model):
    model.eval()

    losses = []
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
            loss = outputs[0]
            losses.append(loss.item())
            logits = outputs[1]
            if args.model_type == 'bert':
                pred = torch.argmax(logits, dim=-1)
            else:
                label_mask = batch['labels'].gt(-1)
                pred = model.crf.decode(logits, mask=label_mask)
                pred = [torch.LongTensor(p) for p in pred]
                pred = pad_sequence(pred, batch_first=True, padding_value=-1)
            
            ref = batch['labels']
            refs.append(ref.view(-1).tolist())
            preds.append(pred.view(-1).tolist())
            

    metric = evaluate.load("seqeval")
    predictions = [
        [args.id2entity[p] for (p, l) in zip(pred, ref) if l != -1]
        for pred, ref in zip(preds, refs)
    ]
    references = [
        [args.id2entity[l] for (p, l) in zip(pred, ref) if l != -1]
        for pred, ref in zip(preds, refs)
    ]
    f1 = metric.compute(predictions=predictions, references=references)['overall_f1']
    loss = sum(losses) / len(losses)
    return {'f1': f1, 'loss': loss}


def train(args, model, tokenizer):
    best_ckpt_name = None
    best_metric = {'f1': -float('inf'), 'loss': float('inf')}
    step = 0
    iteration = 0
    logging.info('=====begin loading dataset====')
    loaders = attach_dataloader(args, tokenizer)
    logging.info('=====end loading dataset====')
    train_dataloader = loaders['train']
    dev_dataloader = loaders['dev']
    
    optimizer = attach_optimizer(args, model)
    scheduler = attach_scheduler(args, optimizer, train_dataloader)
    model.train()

    step_losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    train_iter = trange(args.num_epochs, desc="Epoch", disable=0)
    for epoch in train_iter:
        epoch_iter = tqdm(train_dataloader, desc="Iteration", disable=-1)
        for batch in epoch_iter:
            model.train()
            with torch.cuda.amp.autocast(enabled=args.use_fp16):
                outputs = model(
                    input_ids=batch['input_ids'],
                    token_starts=batch['token_starts'],
                    labels=batch['labels'], 
                    attention_mask=batch['attention_mask'],
                )
                loss = outputs[0]
                scaler.scale(loss).backward()
                step_losses.append(loss.item())

            iteration += 1
            if iteration % args.gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), 
                    max_norm=args.max_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                step += 1
                if step % args.evaluation_steps == 0:
                    metric = validate(args, dev_dataloader, model)
                    best_ckpt_name, best_metric = save_model(best_ckpt_name, metric, best_metric)
                    logging.info('eval f1 : {}'.format(metric['f1']))
                    logging.info('eval loss : {}'.format(metric['loss']))

                    if args.use_wandb:
                        wandb.log({'train loss': sum(step_losses)/len(step_losses), 'step': step})
                        wandb.log({'learning rate': scheduler.get_last_lr()[0], 'step': step})
                        wandb.log({'eval_f1': metric['f1'], 'step': step})
                        wandb.log({'eval_loss': metric['loss'], 'step': step})
                    step_losses = []
                    
    save_final_model(best_ckpt_name)
    return


def ner_pipeline(args, sent, model, tokenizer):
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
    logits = outputs[0]
    entities = []
    words = []
    # A SMALL TRICK FOR TEST
    # since our training data has much denser label
    # while testing data has much sparser label
    # we want to modify the logits to increase the rate of "O" label
    logits[:, :, 0] += 7
    # ==========================
    if args.model_type == 'bert':
        preds = torch.argmax(logits, dim=-1)[0].tolist()
    else:
        preds = model.crf.decode(logits)[0]

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


def inference(args, model, tokenizer):
    def unk_wrapper(word):
        return tokenizer.decode(tokenizer.encode(word), skip_special_tokens=True)

    model.load_state_dict(
        torch.load(
            os.path.join(args.ckpt_save_dir, 'best_{}4{}.ckpt'.format(args.model_type, args.task))
    ))
    with open(args.output_file, 'w', newline='') as output_f, open(args.inference_file, 'r') as input_f:
        sents = input_f.readlines()
        for sent in tqdm(sents):
            words = sent.strip().split(' ')
            src_words, src_entities = ner_pipeline(args, sent, model, tokenizer)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=str, default=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(round(time.time()*1000))/1000)))
    parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'bertbilstmcrf', 'bertcrf'])
    parser.add_argument('--model_name', type=str, default='allenai/scibert_scivocab_uncased', help='model name or path')
    parser.add_argument('--train_file', type=str, default='./data/sciner_dataset/train.conll', help='path to train file, conll for sciner')
    parser.add_argument('--dev_file', type=str, default='./data/sciner_dataset/validation.conll', help='path to dev file')
    parser.add_argument('--test_file', type=str, default='./data/sciner_dataset/validation.conll', help='path to test file')
    parser.add_argument('--inference_file', type=str, default='./data/anlp_test/anlp-sciner-test.txt', help='final ANLP submission file')
    parser.add_argument('--output_file', type=str, default='./data/anlp_test/anlp_haofeiy_sciner.conll')
    parser.add_argument('--task', type=str, default='sciner-finetune', choices=['sciner-finetune'])
    parser.add_argument('--load_from_ckpt', type=str, default=None, help='contine finetuning based on one ckpt')
    parser.add_argument('--model_chosen_metric', type=str, default='f1', help='choose dev ckpt based on this metric')
    parser.add_argument('--ckpt_save_dir', type=str, default='./checkpoints/')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_step', type=int, default=4)
    parser.add_argument('--dev_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--crf_learning_rate', type=float, default=5e-2)
    parser.add_argument('--optimizer_type', type=str, default='adamw')
    parser.add_argument('--scheduler_type', type=str, default='cosine')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, default='sciner')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--evaluation_steps', type=int, default=50)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--max_norm', type=float, default=5.0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('-n', '--id2entity', nargs='+', default=[
        'O',
        'B-MethodName', 'I-MethodName', 'B-HyperparameterName', 'I-HyperparameterName',
        'B-HyperparameterValue', 'I-HyperparameterValue', 'B-MetricName', 'I-MetricName',
        'B-MetricValue', 'I-MetricValue', 'B-TaskName', 'I-TaskName', 'B-DatasetName', 'I-DatasetName',
    ])
    args = parser.parse_args()
    set_seed(args)
    set_wandb(args)


    tokenizer = attach_tokenizer(args)
    model = attach_model(args)

    if args.train:
        train(args, model, tokenizer)
    elif args.inference:
        inference(args, model, tokenizer)
