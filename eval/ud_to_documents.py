# Parses test data inputs and writes plain text and span offset files
# into data/preprocessed/documents.

import json
import re
from pathlib import Path

sent_id_re = re.compile(r'^#\s*sent_id\s*=\s*(.+)\.(\d+)')


def main():
    infile = 'data/turku-ner-corpus/data/UD_Finnish-TDT/fi_tdt-ud-test.conllu'
    outputdir = Path('data/preprocessed/documents')
    outputdir.mkdir(parents=True, exist_ok=True)
    
    lines = open(infile).readlines()
    documents = group_by_documents(lines)
    for document, doc_id in documents:
        text = []
        spans = []
        i = 0
        for sentence in group_by_sentences(document):
            needs_space = False

            sentence = skip_multi_word_tokens(sentence)
            for token in sentence:
                features = token.split('\t')

                if needs_space:
                    text.append(' ')
                    i += 1

                word = features[1]

                spans.append({
                    'token': word,
                    'offset': i,
                })

                text.append(word)
                i += len(word)

                needs_space = 'SpaceAfter=No' not in features[9]

            text.append('\n')
            i += 1

        with open(outputdir / f'{doc_id}.txt', 'w') as outtxt:
            outtxt.write(''.join(text))
        with open(outputdir / f'{doc_id}.spans', 'w') as outspans:
            outspans.write(json.dumps(spans, indent=2, ensure_ascii=False))


def skip_multi_word_tokens(conllu_lines):
    res = []
    skip = []
    for line in conllu_lines:
        features = line.strip().split('\t')
        tid = features[0]
        if '-' in tid:
            start, end = tid.split('-')
            skip = list(range(int(start), int(end) + 1))
            res.append(line)
        else:
            if tid.isdigit() and int(tid) not in skip:
                res.append(line)

    return res


def group_by_sentences(conllu_lines):
    current_sentence = []
    for line in conllu_lines:
        line = line.strip()
        sent_id_match = sent_id_re.match(line)
        if sent_id_match:
            # new sentence start
            if current_sentence:
                yield current_sentence

            current_sentence = []
            continue

        elif not line or line.startswith('#'):
            continue

        else:
            current_sentence.append(line)
        
    if current_sentence:
        yield current_sentence


def group_by_documents(conllu_lines):
    current_doc = []
    current_doc_id = None

    for line in conllu_lines:
        line = line.strip()
        sent_id_match = sent_id_re.match(line)
        if sent_id_match and current_doc_id != sent_id_match.group(1):
            # new document start
            if current_doc:
                yield (current_doc, current_doc_id)

            current_doc = []
            current_doc_id = sent_id_match.group(1)
            continue

        current_doc.append(line)
        
    if current_doc:
        yield (current_doc, current_doc_id)


if __name__ == '__main__':
    main()
