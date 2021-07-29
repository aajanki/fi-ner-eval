import logging
import requests
import sys
from pathlib import Path
from tqdm import tqdm
from .alignment import align_with_ground_truth, merge_ground_truth
from .data import load_documents, load_ground_truth, count_documents, write_tsv3


def main():
    """Predict NER labels on the test set.

    Saves the output in ner_results/turku.conllu"""
    
    exit_if_not_connected()

    doc_dir = Path('data/preprocessed/documents')
    ground_truth_file = Path('data/preprocessed/turku-one/test.tsv')
    output_path = Path('ner_results/turku.conllu')

    documents = load_documents(doc_dir, include_spans=False)
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
        logging.error('Failed to connect to the turku-ner-model. Have you started it on port 8080?')
        sys.exit(1)


if __name__ == '__main__':
    main()
