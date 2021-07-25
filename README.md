# Evaluation of Finnish NER software

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Azure credentials
cp secrets_template.json secrets.json
# Write your Azure endpoint and API key into secrets.json

# Download datasets
git submodule init

# Prepare data
python eval/ud_to_documents.py
```
