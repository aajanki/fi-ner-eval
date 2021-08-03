import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from .conlleval import evaluate, metrics, parse_args


def main():
    df = load_ner_results()

    plot_precision_recall(df)
    plt.savefig('ner_results/prec_rec.png')

    plot_f1(df)
    plt.savefig('ner_results/f1.png')

    print('Result plots saved as ner_results/prec_rec.png and ner_results/f1.png')

    plt.show()


def plot_precision_recall(df):
    plt.figure(figsize=(8, 4.8))
    score_order = ['Organization precision', 'Organization recall',
                   'Person precision', 'Person recall',
                   'GPE precision', 'GPE recall']
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
    plt.figure(figsize=(6.4, 4.8))
    df_f1 = df[df['measure'] == 'f1']
    paired_colors = sns.color_palette("Paired")
    palette_f1 = {
        'Organization': paired_colors[1],
        'Person': paired_colors[3],
        'GPE': paired_colors[5],
    }
    entity_order = ['Organization', 'Person', 'GPE']
    ax = sns.barplot(data=df_f1, x='service', y='score', hue='entity',
                     hue_order=entity_order, palette=palette_f1)
    sns.despine()
    ax.legend(title=None)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.title('F1 score')
    plt.ylim([0, 1])


def load_ner_results():
    services = [
        ('Azure','azure.tsv'),
        ('FiNER', 'finer.tsv'),
        ('Turku NER', 'turku.tsv')
    ]
    interesting_types = ['PERSON', 'ORG', 'GPE']
    entity_name = {
        'PERSON': 'Person',
        'ORG': 'Organization',
        'GPE': 'GPE'
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
