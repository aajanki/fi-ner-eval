import copy
import json
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from pathlib import Path
from tqdm import tqdm
from .data import load_documents_and_ground_truth, write_tsv3


def main():
    data_path = Path('data/preprocessed/documents')
    ground_truth_file = Path('data/turku-ner-corpus/data/conll/test.tsv')
    output_path = Path('ner_results')

    secrets = load_secrets()
    text_analytics_client = ner_client(secrets)

    documents, ground_truth_by_documents = \
        load_documents_and_ground_truth(data_path, ground_truth_file)
    with open(output_path / 'azure.conllu', 'w') as output_f:
        for doc, ground_truth in tqdm(zip(documents, ground_truth_by_documents)):
            spans = doc.pop('spans')
            parts = split_long_document(doc)
            response = text_analytics_client.recognize_entities(parts, language="fi")

            # save the response for debugging purposes
            save_response(response)

            tokens = convert_response_to_conllu(response, parts, spans)
            tokens = append_ground_truth(tokens, ground_truth)
            features = as_conllu_features(tokens)
            write_tsv3(features, output_f)


def load_secrets():
    with open('secrets.json') as f:
        return json.load(f)


def ner_client(secrets):
    credential = AzureKeyCredential(secrets['azure_ner']['api_key'])
    endpoint = secrets['azure_ner']['endpoint']
    return TextAnalyticsClient(endpoint, credential)




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
    output_path = Path('ner_results/azure/responses')
    output_path.mkdir(parents=True, exist_ok=True)

    for res in response:
        if res.is_error:
            print(f'WARNING: error response from Azure on document ID {res.id}')
            
        p = output_path / f'{res.id}.json'
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
    obj = {}
    for key in entity.keys():
        if entity[key]:
            obj[key] = entity[key]
    return obj


def convert_response_to_conllu(response, parts, token_offset, threshold=0.5):
    tokens = copy.copy(token_offset)
    part_offsets = [0] + [len(x['text']) for x in parts][:-1]
    for res, part_offset in zip(response, part_offsets):
        entities = [x for x in res['entities'] if x['confidence_score'] >= threshold]
        for entity in entities:
            offset = part_offset + entity['offset']
            idx = find_matching_tokens(tokens, offset, entity['length'])

            first = True
            for i in idx:
                prefix = 'B-' if first else 'I-'
                entity_code = prefix + entity_short_name(entity)

                if 'entity' in tokens[i]:
                    print(f'WARNING: Duplicate entity: "{tokens[i]["token"]}" '
                          f'at offset {offset} '
                          f'previous = {tokens[i]["entity"]}, new = {entity_code}')

                tokens[i]['entity'] = entity_code

                first = False

    return tokens


def append_ground_truth(tokens, ground_truth):
    assert len(tokens) == len(ground_truth)

    res = []
    for token, gt in zip(tokens, ground_truth):
        assert token['token'] == gt[0]

        token = copy.copy(token)
        token['ground_truth_entity'] = gt[1]
        res.append(token)

    return res


def as_conllu_features(tokens):
    return [(t['token'], t['ground_truth_entity'], t.get('entity', 'O')) for t in tokens]


def entity_short_name(entity):
    short_names = {
        'Location': 'LOC',
        'Person': 'PER',
        'Organization': 'ORG',
    }

    if entity['category'] not in short_names:
        print(f'ERROR: Unknown entity category: {entity["category"]}')

    return short_names[entity['category']]


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


if __name__ == '__main__':
    main()
