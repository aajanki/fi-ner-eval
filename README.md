# Evaluation of Finnish NER software

Scripts for evaluating the precsion and recall of Finnish named entity
recognition (NER) services/libraries.

See [the evaluation results](https://aajanki.github.io/fi-ner-eval/index.html)

## Running the evaluations

### Shared setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install wheel
pip install -r requirements.txt

# Download datasets and models
git submodule update --init --recursive
wget --directory-prefix=/tmp https://korp.csc.fi/download/finnish-tagtools/v1.5/finnish-tagtools-1.5.1.zip
unzip /tmp/finnish-tagtools-1.5.1.zip

# Prepare data
python -m eval.ud_to_documents
python -m eval.turku_one_extract_ud
```

### Azure Cognitive services text analytics NER

```
# Azure credentials
cp secrets_template.json secrets.json

# Deploy an Azure text analytics service and write the endpoint and API key into secrets.json

# Evaluate
python -m eval.ner-azure
python -m eval.conlleval --boundary='-DOCSTART-' --delimiter=$'\t' ner_results/azure.tsv
```

### Turku NER

keras-bert-ner requires Tensorflow 1 which is only available on Python
3.7 or earlier. Therefore, we create a second virtual environment with
Python 3.7 to run the keras-bert-ner server. (TODO: Docker?)

```
eval/download_turku_ner_model.sh

pyenv shell 3.7.7
python3.7 -m venv .venv-keras-bert-ner
source .venv-keras-bert-ner/bin/activate
pip install wheel
pip install -r requirements-keras-bert-ner.txt

python keras-bert-ner/serve.py --ner_model_dir data/models/turku-ner/combined-ext-model
```

Run in the main virtual environment in another terminal window:

```
python -m eval.ner-turku
python -m eval.conlleval --boundary='-DOCSTART-' --delimiter=$'\t' ner_results/turku.tsv
```

### FiNER

```
python -m eval.ner-finer
python -m eval.conlleval --boundary='-DOCSTART-' --delimiter=$'\t' ner_results/finer.tsv
```

### Result plots

Run all the above evaluations first.

```
python -m eval.plot_results
```

### Exploring incorrect predictions

Print tokens with incorrect predictions in the finer results:

```
python -m eval.show_errors < ner_results/finer.tsv | less
```

## Refreshing the report

The Markdown source for the report is located at [docs-source](docs-source) and the generated HTML files at [docs](docs).

Generating the report requires [pandoc-scholar](https://github.com/pandoc-scholar/pandoc-scholar).

```
cp ner_results/*png docs-source/images/ # If the result image have changed

cd docs-source
make
git push  ## Updates the public web pages
```
