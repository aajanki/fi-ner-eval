import argparse
import logging
import subprocess
from pathlib import Path
from tqdm import tqdm
from .alignment import merge_ground_truth
from .data import load_documents, load_ground_truth, count_documents, write_tsv3


def main():
    """Predict NER tags using FiNER."""
    args = parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=getattr(logging, args.loglevel.upper()))

    doc_dir = Path('data/preprocessed/documents')
    ground_truth_file = Path('data/preprocessed/turku-one/test.tsv')
    output_path = Path('ner_results/finer.tsv')

    documents = load_documents(doc_dir, include_spans=False)
    num_documents = count_documents(doc_dir)
    ground_truth_by_documents = load_ground_truth(ground_truth_file)

    with open(output_path, 'w') as output_f:
        for doc, ground_truth in tqdm(zip(documents, ground_truth_by_documents), total=num_documents):
            predicted = predict(doc['text'])
            features = merge_ground_truth(doc['id'], predicted, ground_truth)
            write_tsv3(features, output_f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', default='INFO', help='Set the log level')
    return parser.parse_args()


def predict(text):
    p = subprocess.run('./finnish-nertag', input=text, text=True, capture_output=True,
                       check=True, cwd='finnish-tagtools-1.5.1')
    res = []
    active_chunk = None
    for line in p.stdout.split('\n'):
        if line:
            text, finer_tag = line.split('\t')

            if is_chunk_one(finer_tag):
                assert not active_chunk

                tag = convert_finer_to_ontonotes_tag('B-', finer_tag[1:-2])
            elif is_chunk_start(finer_tag):
                assert not active_chunk

                active_chunk = finer_tag[1:-1]
                tag = convert_finer_to_ontonotes_tag('B-', active_chunk)
            elif is_chunk_end(finer_tag):
                assert active_chunk

                tag = convert_finer_to_ontonotes_tag('I-', active_chunk)
                active_chunk = None
            elif active_chunk:
                tag = convert_finer_to_ontonotes_tag('I-', active_chunk)
            else:
                tag = 'O'

            res.append((text, tag))
    return res


def convert_finer_to_ontonotes_tag(prefix, finer_tag_name):
    tag_map = {
        'EnamexLocPpl': 'GPE',
        'EnamexLocGpl': 'LOC',
        'EnamexLocStr': 'LOC',
        'EnamexLocFnc': 'FAC',
        'EnamexLocAst': 'LOC',
        'EnamexOrgPlt': 'ORG',
        'EnamexOrgClt': 'ORG',
        'EnamexOrgTvr': 'ORG',
        'EnamexOrgFin': 'ORG',
        'EnamexOrgEdu': 'ORG',
        'EnamexOrgAth': 'ORG',
        'EnamexOrgCrp': 'ORG',
        'EnamexPrsHum': 'PERSON',
        'EnamexPrsMyt': 'PERSON',
        'EnamexProXxx': 'PRODUCT',
        'EnamexEvtXxx': 'EVENT',
        'TimexTmeDat': 'DATE',
        'TimexTmeHrm': 'TIME',
        'NumexMsrCur': 'MONEY',
    }
    ignored_names = ['EnamexPrsAnm', 'EnamexPrsTit', 'NumexMsrXxx']

    converted = tag_map.get(finer_tag_name)
    if not converted:
        if finer_tag_name and finer_tag_name not in ignored_names:
            logging.warning(f'Unknown FiNER tag name {finer_tag_name}')

        return 'O'
    else:
        return prefix + converted


def is_chunk_start(finer_tag):
    return len(finer_tag) >= 4 and finer_tag[0] == '<' and finer_tag[1] != '/' and finer_tag[-2] != '/'


def is_chunk_end(finer_tag):
    return len(finer_tag) >= 2 and finer_tag[0] == '<' and finer_tag[1] == '/'


def is_chunk_one(finer_tag):
    return len(finer_tag) >= 2 and finer_tag[-2] == '/' and finer_tag[-1] == '>'


if __name__ == '__main__':
    main()
