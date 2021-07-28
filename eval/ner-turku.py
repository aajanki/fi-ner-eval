import requests
import sys
from pathlib import Path
from tqdm import tqdm
from .data import load_documents, load_ground_truth, count_documents, write_tsv3


def main():
    """Predict NER labels on the test set.

    Saves the output in ner_results/turku.conllu"""
    
    exit_if_not_connected()

    doc_dir = Path('data/preprocessed/documents')
    ground_truth_file = Path('data/preprocessed/turku-one/test.tsv')
    output_path = Path('ner_results/turku.conllu')

    documents = load_documents(doc_dir)
    num_documents = count_documents(doc_dir)
    ground_truth_by_documents = load_ground_truth(ground_truth_file)
    
    with open(output_path, 'w') as output_f:
        for doc, ground_truth in tqdm(zip(documents, ground_truth_by_documents), total=num_documents):
            predicted = predict(doc['text'])
            features = merge_ground_truth(doc['id'], predicted, ground_truth)
            write_tsv3(features, output_f)


def predict(text):
    """Predict NER labels with the keras-bert-ner.

    The server must have been started beforehand on port 8080."""

    endpoint = 'http://localhost:8080'
    data = {'text': text.strip()}
    
    r = requests.get(endpoint, params=data)
    r.raise_for_status()

    tokens = [x.split('\t') for x in r.text.strip('\n').split('\n')]
    return tokens


def exit_if_not_connected():
    try:
        predict('Suomi')
    except requests.exceptions.ConnectionError:
        print('Failed to connect to the turku-ner-model. Have you started it on port 8080?')
        sys.exit(1)


def align_with_ground_truth(docid, predicted, ground_truth, max_look_ahead=9):
    """Align the predicted tokens with the ground truth tokens.

    This is a greedy heuristic: whenever there is a mismatch, it
    skips over tokens until the sequences match again. This happens
    to work on the turku-ner-corpus test test but is no way general.
    It might throw lots of exceptions on another data set with
    different corner cases.

    A proper sequence alignment algorithm might be a good idea..."""

    i = 0 # ground truth index
    j = 0 # predicted index
    aligned = []
    while i < len(ground_truth):
        if ground_truth[i][0] == predicted[j][0]:
            aligned.append([ground_truth[i][0], predicted[j][1]])
            i += 1
            j += 1

        elif i == len(ground_truth) - 1:
            aligned.append([ground_truth[i][0], predicted[j][1]])

            skipped = predicted[j+1:]
            if not label_continues_or_empty([x[1] for x in skipped], predicted[j][1]):
                print(f'WARNING: Discarding predicted entity labels on document {docid}')
                print(skipped)

            i += 1
            j += 1

        elif predicted[j][0].startswith(ground_truth[i][0]):
            # Let's assume the ground truth has multiple tokens
            # corresponding to one predicted token and try to recover.
            next_gt_tokens = [x[0] for x in ground_truth[i+1:i+max_look_ahead]]
            k = next_gt_tokens.index(predicted[j+1][0]) + 1

            predicted_label = predicted[j][1]
            if predicted_label.startswith('B-'):
                continuation_label = 'I-' + predicted_label[2:]
            else:
                continuation_label = predicted_label

            aligned.append([ground_truth[i][0], predicted_label])
            for m in range(1, k):
                aligned.append([ground_truth[i+m][0], continuation_label])

            i += k
            j += 1

        else:
            # predicted has multiple tokens corresponding to one
            # ground truth token.
            next_predicted_tokens = [x[0] for x in predicted[j+1:j+max_look_ahead]]
            k = index_is_starts_of(next_predicted_tokens, ground_truth[i+1][0]) + 1

            predicted_label = predicted[j][1]
            aligned.append([ground_truth[i][0], predicted_label])

            skipped = predicted[j+1:j+k]
            if not label_continues_or_empty([x[1] for x in skipped], predicted_label):
                print(f'WARNING: Discarding predicted entity labels on document {docid}')
                print(skipped)

            i += 1
            j += k

    # Checking the consistency of the entity labels (I-tags only
    # subsequent to the corresponding B-tag) in the output would be a
    # good idea except that the input prediction is often incosistent.

    return aligned


def merge_ground_truth(docid, predicted, ground_truth):
    """Merge predicted and ground truth labels into a combined array.

    The columns of the output are: token, ground truth entity,
    predicted entity."""

    predicted = align_with_ground_truth(docid, predicted, ground_truth)
    assert len(predicted) == len(ground_truth)

    res = []
    for pred, gt in zip(predicted, ground_truth):
        assert pred[0] == gt[0]

        res.append([gt[0], gt[1], pred[1]])

    return res


def index_is_starts_of(arr, key):
    for i, v in enumerate(arr):
        if key.startswith(v):
            return i

    raise ValueError(f"No prefix of '{key}' in list")


def label_continues_or_empty(seq, previous_label):
    """Returns True if seq continues previous_label or 'O' (that is no new
    tags starting)."""

    seq = list(seq) # copy

    if previous_label.startswith('B-'):
        continuation_label = 'I-' + previous_label[2:]
    elif previous_label.startswith('I-'):
        continuation_label = previous_label
    else:
        continuation_label = 'O'

    while seq and seq[0] == continuation_label:
        seq.pop(0)

    return all(x == 'O' for x in seq)


if __name__ == '__main__':
    main()
