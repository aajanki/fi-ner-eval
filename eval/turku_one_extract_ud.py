import re
from itertools import islice
from pathlib import Path
from .data import load_documents, load_ground_truth, write_tsv2


def main():
    input_path = Path('data/turku-one/data/conll/test.tsv')
    output_path = Path('data/preprocessed/turku-one/test.tsv')
    doc_dir = Path('data/preprocessed/documents')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 5

    documents = load_documents(doc_dir)
    tokens = next(load_ground_truth(input_path))

    doc = next(documents)
    next_document_ngram = [x['token'] for x in doc['spans'][:n]]

    current_doc = []
    stop = False
    doc_count = 0
    with open(output_path, 'w') as fp:
        for t, win in zip(tokens, window(tokens, n)):
            ngram = [x[0] for x in win]
            if ngram == next_document_ngram:
                # next document

                if doc['id'] == 's203':
                    # Remove the law documents that follow this
                    # document in turku-one data. The last word really
                    # belonging to this document is "jäsen".
                    last_idx = [x[0] for x in current_doc].index('jäsen')
                    current_doc = current_doc[:last_idx + 1]

                if current_doc:
                    write_tsv2(current_doc, fp)
                    doc_count += 1

                if stop:
                    break

                current_doc = [t]

                try:
                    doc = next(documents)
                    next_document_ngram = [x['token'] for x in doc['spans'][:n]]
                    next_document_ngram = flat_map(retokenize, next_document_ngram)[:n]
                except StopIteration:
                    # This is the last UD document. Next, look for the
                    # start of the first Finer document and stop.
                    next_document_ngram = ['Apple', 'joutumassa', 'veromyrskyn', 'silmään', ':'][:n]
                    stop = True
            else:
                current_doc.append(t)

        if current_doc and not stop:
            write_tsv2(current_doc, fp)
            doc_count += 1

    print(f'Wrote {doc_count} documents into {output_path}')


def retokenize(w):
    """Retokenize w like turku-one has been tokenized."""
    if ' ' in w:
        res = w.split(' ')
    elif '-' in w or '/' in w or ':' in w:
        res = re.split(r'([-/:])', w)
        res = [x for x in res if x]
    elif w.endswith('.') or w.endswith(':'):
        c = w[-1]
        res = []
        while w.endswith(c):
            res.append(c)
            w = w[:-1]
        if w:
            res.insert(0, w)
    elif w == 'Первома́йск':
        res = ['Первома', '́', 'йск']
    else:
        res = [w]
    return res


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


if __name__ == '__main__':
    main()
