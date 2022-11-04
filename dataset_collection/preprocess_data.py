import json
import regex as re


def extract_from_raw(file_path):
    """Read raw data from file_path and extract url, title, abstract and content.
    raw data format:
    {
        "url": ...,
        "title": ...,
        "abstract": ...,
        "sections": [
            {
                "heading": ...,
                "text": ...
            },
            ...
        ]
        ...
    }
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    res = []
    for item in data:
        tmp = {}
        tmp['url'] = item['url']
        tmp['title'] = item['title']
        tmp['abstract'] = [item['abstract']]
        tmp['content'] = []
        for paragraph in item['sections']:
            tmp['content'].append(paragraph['text'])
        res.append(tmp)
    return res

def generate_annotations(spans, texts, label):
    """Generate annotations from spans and texts
    Args:
        spans: a list of tuples, each tuple is the start and end index of the matched result
        texts: a list of strings, each string is the matched result
    Returns:
        A list of annotations, each annotation is a dictionary:
        example:

        [{
            "value": {
                "start": 88,
                "end": 92,
                "text": "0.01",
                "labels": [
                    "MetricValue"
                ]
            },
            "from_name": "label",
            "to_name": "text",
            "type": "labels"
        },]
    """

    annotations = []
    assert len(spans) == len(texts)
    for span, text in zip(spans, texts):
        start, end = span
        annotation = {
            'value': {
                'start': start,
                'end': end,
                'text': text,
                'labels': [label]
            },
            'from_name': 'label',
            'to_name': 'text',
            'type': 'labels'
        }
        annotations.append(annotation)

    return annotations

def span_all(pattern, template, flags=re.IGNORECASE, filter_regex=r'(19|20)\d{2}'):
    """Get all the spans according to the regex pattern from the template string
    Args:
        pattern: regex pattern
        template: string to be matched
        flags: regex flags
        filter_regex: regex pattern to filter out the matched results
    Returns:
        spans: a list of tuples, each tuple is the start and end index of the matched result
        patterns: a list of strings, each string is the matched result
    """
    l = [(0, 0)]
    patterns = []
    match = re.search(pattern, template, flags)
    while match:
        span1 = match.span()
        res = match.group()
        span2 = (span1[0] + l[-1][1], span1[1] + l[-1][1])
        if not re.search(filter_regex, res):
            l.append(span2)
            patterns.append(res)
        template = template[span1[1]:]
        match = re.search(pattern, template, flags)
    return l[1:], patterns

def annotate_numbers(paragraph):
    """Annotate all the numbers with regex
    Args:
        paragraph: string to be annotated
    Returns:
        A list of annotations, each annotation is a dictionary
    """

    metric_regex = r'(?<!table\s|fig\s|figure\s|section\s|\d.)\d[.]?[^a-z\s][\d]*[%]?'
    filter_regex=r'(19|20)\d{2}'
    spans, numbers = span_all(metric_regex, paragraph, re.IGNORECASE, filter_regex)

    parameter_regex = r'\d+[e][-]?[\d]*'
    spans_para, parameters = span_all(parameter_regex, paragraph, re.IGNORECASE, filter_regex)

    annotate_numbers = generate_annotations(spans, numbers, 'MetricValue')
    annotate_parameters = generate_annotations(spans_para, parameters, 'HyperparameterValue')
    # merge para and number
    for item in annotate_parameters:
        if item['value']['text'] not in numbers:
            annotate_numbers.append(item)
        else:
            index = numbers.index(item['value']['text'])
            annotate_numbers[index]['value']['labels'].append('HyperparameterValue')

    return annotate_numbers



if __name__ == "__main__":
    for sentence in [
            "Figure 1: Denotation experiment finds the best input setting for data collection, that preserves meaning but diversifies styles among annotators with different personas.",
            "FiD-base and FiD-large contain L = 12 and 24 layers respectively, and we set the budget B = Lk.",
            "We consider various regularization parameters for SVM and logistic regression (e.g., c=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0].",
            "The learning rate is 1e-4 and the batch size is 32.",
            "We have 1 term, ",
        ]:
        annotated_data = annotate_numbers(sentence)
        print(annotated_data)

    