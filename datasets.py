def load_ADR_lexicon(path):
    results = []
    with open(path) as f:
        for i in xrange(21):
            f.readline()
        for line in f:
            results.append(line.split('\t'))
    return dict(zip(('id', 'phrase', 'source'), zip(*results)))


