import sys
from collections import Counter


class ErrorInstances():
    def __init__(self):
        self.fp_by_type = {}
        self.fn_by_type = {}


def main():
    interesting_types = ['PERSON', 'LOC', 'GPE', 'ORG', 'EVENT', 'PRODUCT']
    errors = ErrorInstances()
    
    tokens = load_ner_results(sys.stdin)
    for features in tokens:
        text = features[0]
        correct = features[1]
        predicted = features[2]

        if entity_type(correct) != entity_type(predicted):
            for t in interesting_types:
                if entity_type(correct) == t:
                    errors.fn_by_type.setdefault(t, []).append((text, entity_type(predicted)))
                elif entity_type(predicted) == t:
                    errors.fp_by_type.setdefault(t, []).append((text, entity_type(correct)))


    for t in interesting_types:
        print(f'----- {t} -----\n')
        print(f'Correct type is {t} but was predicted as something else:')
        for (text, predicted), freq in sorted(Counter(errors.fn_by_type.get(t, [])).items(), key=lambda x: x[1]):
            print(f'{freq:<3} {text} {predicted}')
        print()

        print(f'Predicted {t} but should have been something else:')
        for (text, correct), freq in sorted(Counter(errors.fp_by_type.get(t, [])).items(), key=lambda x: x[1]):
            print(f'{freq:<3} {text} {correct}')
        print()


def load_ner_results(fp):
    for line in fp:
        line = line.rstrip('\n')
        if line:
            yield line.split('\t')


def entity_type(x):
    if x == 'O':
        return x
    else:
        return x[2:]


def same_entity_type(a, b):
    return entity_type(a) == entity_type(b)


if __name__ == '__main__':
    main()
