import json


def get_sent_examples_from_bio_fpath(config, bio_fpath, evaluate):
    sents = []
    sent_tags = []
    with open(bio_fpath) as infile:
        for line in infile:
            array = line.split('\t')
            assert len(array) >= 2
            sents.append(array[0])
            sent_tags.append(array[-1].replace('\n', ''))

    if not evaluate:
        vocab = {tag: index for index, tag in enumerate(set(sent_tags))}
        with open(config.vocab_fpath, 'w') as f:
            json.dump(vocab, f)
    return [{'words': sent.split(), 'label': label} for sent, label in zip(sents, sent_tags)]
