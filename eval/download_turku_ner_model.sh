#!/bin/sh

set -eu

# Download the trained model
mkdir -p data/models/turku-ner
wget --directory-prefix=data/models/turku-ner http://dl.turkunlp.org/turku-ner-models/combined-ext-model-130220.tar.gz
(cd data/models/turku-ner; tar xzvf combined-ext-model-130220.tar.gz)
