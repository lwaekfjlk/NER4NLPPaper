# Scientific Entity Recognition for NLP Conference Paper

The project is related to [nlp-from-scratch-assignment-2022](https://github.com/neubig/nlp-from-scratch-assignment-2022). 

## Task

This project is working on the task of *scientific entity recognition*, specifically in the domain of NLP papers from recent NLP conferences (e.g. ACL, EMNLP, and NAACL). Specifically, we try to identify entities such as task names, model names, hyperparameter names and their values, and metric names and their values in these papers.

7 kinds of entities including seven varieties of entity: `MethodName`, `HyperparameterName`, `HyperparameterValue`, `MetricName`, `MetricValue`, `TaskName`, and `DatasetName` are defined as scientific entities in this task. More information related to the annotation of entities can be seen in [annotation standard](https://github.com/neubig/nlp-from-scratch-assignment-2022/blob/main/annotation_standard.md).

**Input:** The input to our model will be a text file with one paragraph per line. The text will *already be tokenized* using the [spacy](https://spacy.io/api/tokenizer/) tokenizer, and you should not change the tokenization. An example of the input looks like this:

```
Recent evidence reveals that Neural Machine Translation ( NMT ) models with deeper neural networks can be more effective but are difficult to train .
```

**Output:** The output of our model should be a file in [CoNLL format](https://simpletransformers.ai/docs/ner-data-formats/#text-file-in-conll-format), with one token per line, a tab, and then a corresponding tag.

**Evaluation:** The outputs of our model are evaluated through the Explainaboard. You will also want to install the latest version of [explainaboard_client](https://github.com/neulab/explainaboard_client) which will be used for system submission, and can also be used for your own evaluation if you choose.

## Dataset

We annotated 807 sentences by ourselves, including 686 train sentences and 121 test sentences.

We tested on the officially released test data [here](https://github.com/neubig/nlp-from-scratch-assignment-2022/tree/main/data) via ExplainaBoard.

The whole dataset creation pipeline can be divided into three stages: raw data collection, sentence pre-annotation and selection, and human annotation. Details and implementation can be seen in the [dataset_collection](https://github.com/lwaekfjlk/NER4NLPPaper/tree/main/dataset_collection) part.

| Entity Type | `DatasetName` | `TaskName` | `MethodName` | `MetricName` | `MetricValue` | `HyperparameterName` | `HyperparameterValue` |
| ----------- | ------------- | ---------- | ------------ | ------------ | ------------- | -------------------- | --------------------- |
| #Total      | 430           | 347        | 601          | 254          | 266           | 300                  | 342                   |

## Code

We provide 4 types of NER implementation for this task:

1. Huggingface official NER pipeline
2. BERT
3. BERT+CRF
4. BERT+BiLSTM+CRF

We reached an F1 score of **50.27** on the test data.

For fine-tuning on our annotated dataset:

1. Huggingface official NER pipeline fine-tuning

```bash
cd ./ner_huggingface
sh run_sciner_finetune.sh
```

2. BERT

```bash
cd ./ner_baseline/run_sh
sh run_bert_sciner-finetune.sh
```

3. BERT+CRF

```bash
cd ./ner_baseline/run_sh
sh run_bertcrf_sciner-finetune.sh
```

4. BERT+BiLSTM+CRF

```bash
cd ./ner_baseline/run_sh
sh run_bertbilstmcrf_sciner-finetune.sh
```

For testing on our validation data and official test data:

```bash
cd ./ner_baseline/run_sh
sh run_sciner-test-inference.sh
sh explainaboard_test_submit.sh
sh run_sciner-valid-inference.sh
sh explainaboard_valid_submit.sh
```

Such command enables you to generate the submitted form of CONLL file and submit your results to the ExplainaBoard.

The basic usage of ExplainaBoard is shown below (please make sure you use the latest version of `explainaboard_client` otherwise might cause warning and error):

```
pip install --upgrade --force-reinstall explainaboard_client==0.0.11 (should the latest version) explainaboard_api_client==0.2.8
```

Set the username and API key for the client:

```
export EB_USERNAME="[your username]"
export EB_API_KEY="[your API key]"
```

More information related to the submission and evaluation can be checked through [here](https://github.com/neubig/nlp-from-scratch-assignment-2022/blob/main/evaluation_and_submission.md).

