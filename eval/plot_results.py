import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from .conlleval import evaluate, metrics, parse_args
from .functools import flat_map

entity_plot_order = ['Product', 'Event', 'Organization', 'Person', 'GPE', 'Location']


def main():
    matplotlib.rcParams.update({'font.size': 14})
    
    df = load_ner_results()

    plot_precision_recall(df)
    plt.savefig('ner_results/prec_rec.png', dpi=72)

    plot_f1(df)
    plt.savefig('ner_results/f1.png', dpi=72)

    print('Result plots saved as ner_results/prec_rec.png and ner_results/f1.png')

    plt.show()


def plot_precision_recall(df):
    plt.figure(figsize=(9, 4.8))
    score_order = flat_map(lambda x: [x + ' precision', x + ' recall'], entity_plot_order)
    paired_colors = sns.color_palette("Paired")
    palette = {measure: color
               for (measure, color) in zip(score_order, paired_colors)}
    sns.barplot(data=df, x='service', y='score', hue='entity_measure',
                hue_order=score_order, palette=palette)
    sns.despine()
    plt.legend(title=None, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title('Precision and recall')
    plt.ylim([0, 1])
    plt.tight_layout()


def plot_f1(df):
    plt.figure(figsize=(9, 4.8))
    df_f1 = df[df['measure'] == 'f1']
    paired_colors = sns.color_palette("Paired")
    palette_f1 = {
        entity: paired_colors[2*i + 1] for i, entity in enumerate(entity_plot_order)
    }
    ax = sns.barplot(data=df_f1, x='service', y='score', hue='entity',
                     hue_order=entity_plot_order, palette=palette_f1)
    sns.despine()
    ax.legend(title=None, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title('F1 score')
    plt.ylim([0, 1])
    plt.tight_layout()


def load_ner_results():
    services = [
        ('Azure','azure.tsv'),
        ('FiNER', 'finer.tsv'),
        ('Turku NER', 'turku.tsv')
    ]
    interesting_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']
    entity_name = {
        'PERSON': 'Person',
        'ORG': 'Organization',
        'GPE': 'GPE',
        'LOC': 'Location',
        'PRODUCT': 'Product',
        'EVENT': 'Event',
    }

    eval_args = parse_args(['--boundary=-DOCSTART-', '--delimiter=\t'])
    data = []
    for service_name, result_file in services:
        with open(Path('ner_results') / result_file) as f:
            counts = evaluate(f, eval_args)
        overall, by_type = metrics(counts)

        for ne_type in interesting_types:
            data.append({
                'service': service_name,
                'entity': entity_name[ne_type],
                'measure': 'precision',
                'score': by_type[ne_type].prec
            })
            data.append({
                'service': service_name,
                'entity': entity_name[ne_type],
                'measure': 'recall',
                'score': by_type[ne_type].rec
            })
            data.append({
                'service': service_name,
                'entity': entity_name[ne_type],
                'measure': 'f1',
                'score': by_type[ne_type].fscore
            })

    df = pd.DataFrame(data)
    df['entity_measure'] = df['entity'] + ' ' + df['measure']
    return df


if __name__ == '__main__':
    main()
