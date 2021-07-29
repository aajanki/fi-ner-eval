import json


def load_documents(doc_dir):
    for p_txt in sorted(doc_dir.glob('*.txt')):
        docid = p_txt.stem
        p_spans = p_txt.with_suffix('.spans')
        with p_txt.open() as f_txt, p_spans.open() as f_spans:
            yield {'id': docid, 'text': f_txt.read(), 'spans': json.load(f_spans)}


def count_documents(doc_dir):
    return len(list(doc_dir.glob('*.txt')))


def load_ground_truth(path):
    current_document = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            features = line.split('\t')
            if features[0] == '-DOCSTART-':
                # new document starts
                if current_document:
                    yield current_document

                current_document = []
            else:
                current_document.append(features)

    if current_document:
        yield current_document


def write_tsv2(tokens, fp):
    fp.write('-DOCSTART-\tO\n')
    for (text, entity) in tokens:
        fp.write(f'{text}\t{entity}\n')


def write_tsv3(tokens, fp):
    fp.write('-DOCSTART-\tO\tO\n')
    for (text, grount_truth_entity, predicted_entity) in tokens:
        fp.write(f'{text}\t{grount_truth_entity}\t{predicted_entity}\n')
