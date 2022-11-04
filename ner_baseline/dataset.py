import jsonlines
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SciNERDataset(Dataset):
    def __init__(self, args, tokenizer, split, sep=' -X- _ '):
        self.tokenizer = tokenizer
        self.entity2id = {e: i for i, e in enumerate(self.args.id2entity)}
        self.id2entity = {i: e for i, e in enumerate(self.args.id2entity)}
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.pad = self.tokenizer.pad_token_id
        file_dict = {'train': args.train_file, 'dev': args.dev_file, 'test': args.test_file}
        self.data = self.collect_data(file_dict[split], sep)


    def collect_data(self, file, sep):
        data = []
        sent = []
        with open(file, 'r') as f:
            lines = f.readlines()
        for l in lines:
            l = l.strip()
            if len(l) == 0:
                instance = self.collect_instance(sent, sep=sep)
                data.append(instance)
                sent = []
            else:
                sent.append(l)
        return data

    def collect_instance(self, conll_sentence, sep=' -X- _ '):
        instance = {'input_ids': [], 'token_starts': [], 'labels': []}
        for line in conll_sentence:
            token, tag = line.split(sep)
            token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))
            label = self.entity2id[tag]
            labels = [self.entity2id[tag]]
            token_starts = [1] + [0 for _ in range(len(token_ids) - 1)]
            instance['input_ids'] += token_ids
            instance['token_starts'] += token_starts
            instance['labels'] += labels
        instance['input_ids'] = [self.cls] + instance['input_ids'] + [self.sep]
        instance['token_starts'] = [0] + instance['token_starts'] + [0]
        return instance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch, args):
        max_len = args.max_length
        device = args.device

        input_ids = []
        token_starts = []
        labels = []
        crf_labels = []
        attention_mask = []

        for instance in batch:
            input_id = torch.LongTensor(instance['input_ids'])
            token_start = torch.ByteTensor(instance['token_starts'])
            label = torch.LongTensor(instance['labels'])
            input_ids.append(input_id)
            token_starts.append(token_start)
            labels.append(label)
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad)
        token_starts = pad_sequence(token_starts, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        attention_mask = torch.ne(input_ids, self.pad)

        return {
            'input_ids': input_ids[:, :max_len].to(device),
            'token_starts': token_starts[:, :max_len].to(device),
            'labels': labels[:, :max_len].to(device),
            'attention_mask': attention_mask[:, :max_len].to(device),
        }

