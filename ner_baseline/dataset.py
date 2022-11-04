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
        if split == 'test':
            file = args.test_file
        elif split == 'dev':
            file = args.dev_file
        elif split == 'train':
            file = args.train_file
        self.data = self._read_conll_file(file, sep)


    def _read_conll_file(self, file, sep):
        with open(file, 'r') as f:
            lines = f.readlines()

        data = []
        sent = []

        for l in lines:
            l = l.strip()
            if len(l) == 0:
                instance = self._create_example(sent, sep=sep)
                data.append(instance)
                sent = []
            else:
                sent.append(l)
        return data

    def _create_example(self, conll_sentence, sep=' -X- _ '):
        example = {'input_ids': [], 'token_starts': [], 'labels': []}
        for line in conll_sentence:
            token, tag = line.split(sep)
            token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(token))
            label = self.entity2id[tag]
            labels = [self.entity2id[tag]]
            token_starts = [1] + [0 for _ in range(len(token_ids) - 1)]
            example['input_ids'] += token_ids
            example['token_starts'] += token_starts
            example['labels'] += labels
        example['input_ids'] = [self.cls] + example['input_ids'] + [self.sep]
        example['token_starts'] = [0] + example['token_starts'] + [0]
        return example

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


class ScirexDataset(Dataset):
    # Material here means dataset!!!
    entities = [
        'O',
        'B-Material', 'I-Material', 'B-Metric', 'I-Metric',
        'B-Task', 'I-Task', 'B-Method', 'I-Method'
    ]
    entity2id = {e: i for i, e in enumerate(entities)}
    id2entity = {i: e for i, e in enumerate(entities)}

    def __init__(self, file, tokenizer):
        self.tokenizer = tokenizer
        self.data = self._read_scirex_file(file)

    def _read_scirex_file(self, file):
        data = []
        with jsonlines.open(file) as reader:
            for document in reader:
                data += self.create_examples(document)
        return data

    def create_examples(self, scirex_document):
        res = []
        words = scirex_document['words']

        ner_i = 0
        ner_start, ner_end, entity = scirex_document['ner'][ner_i][0], \
                                     scirex_document['ner'][ner_i][1], \
                                     scirex_document['ner'][ner_i][2]

        for sentence in scirex_document['sentences']:
            instance = {
                'input_ids': [],
                'labels': []
            }
            sen_start, sen_end = sentence[0], sentence[1]
            for word_i in range(sen_start, sen_end):
                token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(words[word_i]))
                if word_i < ner_start:
                    tag = 'O'
                elif word_i == ner_start:
                    tag = 'B-' + entity
                else:
                    tag = 'I-' + entity

                if word_i == ner_end - 1:
                    ner_i += 1
                    if ner_i >= len(scirex_document['ner']):
                        ner_start = float('inf')
                    else:
                        ner_start, ner_end, entity = scirex_document['ner'][ner_i][0], \
                                                     scirex_document['ner'][ner_i][1], \
                                                     scirex_document['ner'][ner_i][2]
                instance['input_ids'] += token_ids
                if len(token_ids) == 1:
                    instance['labels'] += [self.entity2id[tag]]
                else:
                    instance['labels'] += [self.entity2id[tag]] + [-100 for _ in range(len(token_ids)-1)]
            res.append(instance)
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch, max_length=None):
        batch_max_length = max([len(instance['input_ids']) for instance in batch])

        if max_length is None:
            max_length = batch_max_length + 2
        else:
            max_length = min(max_length, batch_max_length + 2)

        input_ids = []
        labels = []
        attention_mask = []

        for instance in batch:
            instance_token_ids = [self.tokenizer.cls_token_id]
            instance_token_ids += instance['input_ids']
            instance_token_ids = instance_token_ids[:(max_length - 1)]
            instance_token_ids += [self.tokenizer.sep_token_id]
            instance_token_ids += [self.tokenizer.pad_token_id for _ in range(max_length - len(instance_token_ids))]
            input_ids.append(instance_token_ids)

            instance_labels = [-100]
            instance_labels += instance['labels']
            instance_labels = instance_labels[:(max_length - 1)]
            instance_labels += [-100 for _ in range(max_length - len(instance_labels))]
            labels.append(instance_labels)

            valid_token_length = min(max_length, len(instance['input_ids']) + 2)
            att_mask = [1 for i in range(valid_token_length)] + [0 for i in range(max_length - valid_token_length)]
            attention_mask.append(att_mask)

        return {
            'input_ids': torch.LongTensor(input_ids),
            'labels': torch.LongTensor(labels),
            'attention_mask': torch.LongTensor(attention_mask)
        }


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, SequentialSampler
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_file = '../data/train.conll'
    train_dataset = ConLLDataset(train_file, tokenizer)
    # train_file = '../data/scirex/train.jsonl'
    # train_dataset = ScirexDataset(train_file, tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=2,
                                  sampler=SequentialSampler(train_dataset),
                                  collate_fn=train_dataset.collate_fn)

    for data in train_dataloader:
        input_ids = data['input_ids']
        labels = data['labels']
        mask = data['attention_mask']
        # if input_ids.shape[1] != labels.shape[1] or labels.shape[1] != mask.shape[1]:
        print(input_ids)
        print(labels)
        print(mask)
        break
