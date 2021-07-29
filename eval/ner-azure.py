import argparse
import copy
import json
import logging
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from pathlib import Path
from tqdm import tqdm
from .alignment import align_with_ground_truth, merge_ground_truth
from .data import load_documents, count_documents, load_ground_truth, write_tsv3

cache_dir = Path('ner_results/azure/responses')


def main():
    args = parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=getattr(logging, args.loglevel.upper()))

    doc_dir = Path('data/preprocessed/documents')
    ground_truth_file = Path('data/preprocessed/turku-one/test.tsv')
    output_path = Path('ner_results')

    secrets = load_secrets()
    text_analytics_client = ner_client(secrets)

    documents = load_documents(doc_dir)
    num_documents = count_documents(doc_dir)
    ground_truth_by_documents = load_ground_truth(ground_truth_file)
    with open(output_path / 'azure.conllu', 'w') as output_f:
        for doc, ground_truth in tqdm(zip(documents, ground_truth_by_documents), total=num_documents):
            if args.cached_response:
                response = predict_cached(doc)
            else:
                response = predict(text_analytics_client, doc)

            # First, align entities with the input tokens using the
            # known offsets
            predicted = align_with_input(doc, response)

            # This helps alignment with the turku-one ground truth
            predicted = expand_ellipses(predicted)

            # Next, sequence align the input tokens with the ground
            # truth tokens
            features = merge_ground_truth(doc['id'], predicted, ground_truth)

            write_tsv3(features, output_f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached-response', action='store_true',
                        default=False,
                        help='Use cached results instead of calling the Azure cloud API')
    parser.add_argument('--loglevel', default='INFO', help='Set the log level')
    return parser.parse_args()


def load_secrets():
    with open('secrets.json') as f:
        return json.load(f)


def ner_client(secrets):
    credential = AzureKeyCredential(secrets['azure_ner']['api_key'])
    endpoint = secrets['azure_ner']['endpoint']
    return TextAnalyticsClient(endpoint, credential)


def predict(client, doc):
    parts = split_long_document(doc)
    response = client.recognize_entities(parts, language="fi")

    # cache the response for debugging purposes
    save_response(response)

    return response


def predict_cached(doc):
    response = []
    for p in cache_dir.glob(f'{doc["id"]}*.json'):
        with p.open() as fp:
            response.append(json.load(fp=fp))
    return response


def split_long_document(doc):
    max_azure_document_length = 5120

    text = doc['text']
    if len(text) > max_azure_document_length:
        parts = []
        while len(text) > max_azure_document_length:
            i = text[:max_azure_document_length].rfind('\n')
            if i >= 0:
                prefix = text[:i+1]
            else:
                prefix = text[:max_azure_document_length]

            docid = f'{doc["id"]}.{len(parts) + 1}'
            parts.append({'id': docid, 'text': prefix})

            text = text[len(prefix):]

        docid = f'{doc["id"]}.{len(parts) + 1}'
        parts.append({'id': docid, 'text': text})
        return parts
    else:
        return [doc]


def save_response(response):
    cache_dir.mkdir(parents=True, exist_ok=True)

    for res in response:
        if res.is_error:
            logging.warning(f'error response from Azure on document ID {res.id}')
            
        p = cache_dir / f'{res.id}.json'
        with p.open('w') as fp:
            resobj = entities_result_as_py_object(res)
            json.dump(resobj, fp=fp, indent=2, ensure_ascii=False)


def entities_result_as_py_object(result):
    if result.is_error:
        obj = {
            'id': result.id,
            'error': simple_azure_response_as_py_object(result.error),
            'is_error': True,
        }
    else:
        entities = [simple_azure_response_as_py_object(e) for e in result.entities]
        warnings = [simple_azure_response_as_py_object(w) for w in result.warnings]
        obj = {
            'id': result.id,
            'entities': entities,
            'warnings': warnings,
            'is_error': False,
        }

    return obj


def simple_azure_response_as_py_object(entity):
    return {
        key: val for key, val in entity.items() if val is not None
    }


def as_conllu_features(tokens):
    return [(t['token'], t['ground_truth_entity'], t.get('entity', 'O')) for t in tokens]


def ontonotes_entity_name(entity):
    names = {
        'Person': 'PERSON',
        'Organization': 'ORG',
    }

    category = entity['category']
    if category == 'Location':
        if entity.get('subcategory') == 'GPE':
            return 'GPE'
        else:
            return 'LOC'
    elif category in names:
        return names.get(category, category)
    else:
        logging.warning(f'Unknown category {category}')
        return category


def find_matching_tokens(tokens, entity_offset, entity_length):
    matches = []
    for i, token in enumerate(tokens):
        one_past_token_end = token['offset'] + len(token['token'])
        one_past_entity_end = entity_offset + entity_length
        if (entity_offset >= token['offset']
            and (entity_offset < one_past_token_end)):
            matches.append(i)
        elif ((entity_offset < token['offset'])
              and (one_past_entity_end >= one_past_token_end)):
            matches.append(i)

    return matches


def align_with_input(input_document, response):
    tokens = tokens_with_entity_codes(input_document, response)
    return tokens_as_conllu_features(tokens)


def tokens_with_entity_codes(input_document, response, threshold=0.5):
    parts = split_long_document(input_document)
    merged_response = merge_response_parts(response, parts)
    tokens = copy.copy(input_document['spans'])
    for ent in merged_response['entities']:
        if ent.get('confidence_score') is None:
            logging.warning(f'confidence_score missing on entity "{ent.get("text")}", '
                            f'document {input_document["id"]}')

        if ent.get('confidence_score', 0.0) > threshold:
            idx = find_matching_tokens(tokens, ent['offset'], ent['length'])

            prefix = 'B-'
            for i in idx:
                entity_code = prefix + ontonotes_entity_name(ent)

                if 'entity' in tokens[i] and tokens[i]['entity'] != entity_code:
                    logging.warning(f'Duplicate entity for token "{tokens[i]["token"]}" '
                                    f'at offset {ent["offset"]} of document {input_document["id"]}, '
                                    f'previous = {tokens[i]["entity"]}, new = {entity_code}')

                tokens[i]['entity'] = entity_code

                prefix = 'I-'

    return tokens


def tokens_as_conllu_features(tokens):
    return [(t['token'], t.get('entity', 'O')) for t in tokens]


def merge_response_parts(response, parts):
    merged_entities = []
    part_offsets = [0] + [len(x['text']) for x in parts][:-1]
    for response_part, part_offset in zip(response, part_offsets):
        assert not response_part['is_error']

        for ent in response_part['entities']:
            ent_corrected = copy.copy(ent)
            ent_corrected['offset'] = part_offset + ent.get('offset', 0)
            merged_entities.append(ent_corrected)

    return {
        'id': response[0]['id'],
        'entities': merged_entities
    }


def expand_ellipses(tokens):
    """Replace "..." token with three "." tokens.
   
    Turku-one tokenizes ellipses as three periods. Re-tokenizing input
    in the same way makes sequence alignment heuristic work better."""
    res = []
    for t in tokens:
        if t[0] == '...':
            res.append(('.', t[1]))
            res.append(('.', t[1]))
            res.append(('.', t[1]))
        else:
            res.append(t)
    return res


if __name__ == '__main__':
    main()
