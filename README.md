# Evaluation of Finnish NER software

Scripts for evaluating the precsion and recall of Finnish named entity
recognition (NER) services/libraries.

## Results

TODO...

## Running the evaluations

### Shared setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install wheel
pip install -r requirements.txt

# Download datasets and models
git submodule update --init --recursive
wget --directory-prefix=tmp https://korp.csc.fi/download/finnish-tagtools/v1.3/finnish-tagtools-1.3.2.zip
unzip tmp/finnish-tagtools-1.3.2.zip

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
python data/turku-ner-corpus/scripts/conlleval.py --boundary='-DOCSTART-' --delimiter=$'\t' ner_results/azure.tsv
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
python data/turku-ner-corpus/scripts/conlleval.py --boundary='-DOCSTART-' --delimiter=$'\t' ner_results/turku.tsv
```

## FiNER

```
python -m eval.ner-finer
python data/turku-ner-corpus/scripts/conlleval.py --boundary='-DOCSTART-' --delimiter=$'\t' ner_results/finer.tsv
```
