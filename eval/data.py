import json


def load_documents_and_ground_truth(doc_dir, ground_truth_file):
    documents = sorted(load_documents(doc_dir), key=lambda x: x['id'])
    ground_truth_by_documents = load_ground_truth(ground_truth_file)
    ground_truth_by_documents = (align_ground_truth(a, b)
                                 for a, b in zip(ground_truth_by_documents, documents))

    return (documents, ground_truth_by_documents)


def load_documents(doc_dir):
    for p_txt in doc_dir.glob('*.txt'):
        docid = p_txt.stem
        p_spans = p_txt.with_suffix('.spans')
        with p_txt.open() as f_txt, p_spans.open() as f_spans:
            yield {'id': docid, 'text': f_txt.read(), 'spans': json.load(f_spans)}


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


def align_ground_truth(ground_truth_document, input_document):
    # The ground truth and the input data have some differences in
    # tokenization. Some tokens in the input data contain whitespace,
    # for example a smiley ": )" and phone numbers, while ground truth
    # has split everything by space. Merge back split ground truth
    # tokens.
    i = 0
    aligned = []
    for token in input_document['spans']:
        input_token = token['token']
        if ' ' in input_token:
            subtokens = input_token.split(' ')
            fixed = [input_token] + ground_truth_document[i][1:]

            skipped_gt_labels = [x[1] for x in ground_truth_document[i+1:i+len(subtokens)]]
            if not all(x == 'O' for x in skipped_gt_labels):
                print(f'WARNING: ignoring an entity label when aligning ground truth tokens '
                      f'on document {input_document["id"]}.')
                for x in ground_truth_document[i+1:i+len(subtokens)]:
                    print(x)
            
            aligned.append(fixed)
            i += len(subtokens)
        else:
            assert ground_truth_document[i][0] == input_token
            
            aligned.append(ground_truth_document[i])
            i += 1

    assert i == len(ground_truth_document)
    return aligned


def write_conllu_tokens(tokens, fp):
    fp.write('-DOCSTART-\tO\tO\n')
    for (text, grount_truth_entity, predicted_entity) in tokens:
        fp.write(f'{text}\t{grount_truth_entity}\t{predicted_entity}\n')
