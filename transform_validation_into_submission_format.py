
with open('./data/sciner_dataset/validation.conll', 'r') as input_f, \
     open('./data/anlp_valid/anlp-sciner-valid-annotated.conll', 'w') as output_conll, \
     open('./data/anlp_valid/anlp-sciner-valid-sentences.txt', 'w') as output_txt:
    word_list = []
    sentence_list = []
    lines = input_f.readlines()
    for idx, line in enumerate(lines):
        lines[idx] = lines[idx].replace(' -X- _ ', '\t')
        if line == '\n':
            word_list.append('\n')
            sentence_list.append(' '.join(word_list))
            word_list = []
        else:
            word_list.append(line.split(' ')[0])

    output_conll.writelines(lines)
    output_txt.writelines(sentence_list)