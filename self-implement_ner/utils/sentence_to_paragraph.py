import argparse

if __name__ == '__main__':
    '''
    This script takes your model outputs (in CoNLL format) on
    sentence-level data (generated by paragraph_to_sentence.py), and
    removes the extra newlines to match the paragraph-segmented outputs
    that we expect.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', default='../data/anlp_test/anlp-sciner-test-sentences.conll',
        help='Your model outputs on the sentence-segmented test set in CoNLL format')
    parser.add_argument('-d', '--dummyfile', default='../data/anlp_test/anlp-sciner-test-empty.conll',
        help='The provided empty CoNLL file, used to determine the original paragraph boundaries')
    parser.add_argument('-o', '--outfile', default='../data/anlp_test/output.conll',
        help='The output CoNLL file that restores paragraph segmentation')
    args = parser.parse_args()

    input_lines = []
    dummy_lines = []
    output_lines = []

    with open(args.infile) as f:
        input_lines = f.readlines()

    with open(args.dummyfile) as f:
        dummy_lines = f.readlines()

    i = 0
    j = 0
    while i < len(input_lines):
        # skip whitespace if dummy file doesn't have it
        if input_lines[i] == '\n' and dummy_lines[j] != '\n':
            i += 1

        output_lines.append(input_lines[i])
        i += 1
        j += 1

    # make sure all tokens match our dummy reference CoNLL file
    assert(len(dummy_lines) == len(output_lines))
    for i in range(len(output_lines)):
        dummy_token = dummy_lines[i].split('\t')[0]
        output_token = output_lines[i].split('\t')[0]
        assert(dummy_token == output_token)

    with open(args.outfile, 'w') as f:
        f.write(''.join(output_lines))