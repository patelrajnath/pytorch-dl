## Pytorch-dl (Deep Learning with Pytorch)
This project implements the classification task using Transformer model. On IMDB sentiment analysis task it achieved a score of 85+ accuracy.

It also contains BERT training- 
* Transformer based Neural MT training and decoding
* Training and fine tuning mBart for Neural MT
* Bert encoder ([Default Bert](https://arxiv.org/pdf/1810.04805.pdf))
* Bert encoder-decoder ([mBart](https://arxiv.org/pdf/2001.08210.pdf))

## Prerequisite
- python (3.6+)
- [pytorch (1.3+)](https://pytorch.org/get-started/locally/)
- [Sentencepiece](https://github.com/google/sentencepiece)
- numpy

# Quick Start
### INSTALL Dependencies
```bash
pip3 install -r requirements.txt
python -m spacy download en
```

### Train NMT model

##### Prepare data
```bash
cd examples/translation/
bash prepare-iwslt14.sh
cd -
bash prep.sh
```

##### Train model
```bash
bash train.sh
```
##### Decode the binarized validation data
```bash
bash decode.sh
```

##### Translate a text file
```bash
bash translate_file.sh
```

### IMDB classification:
```bash
python3 classify.py
```

### Bert training:
```bash
python3 pretrain_bert.py
```

### mBART training:
```bash
python3 pretrain_mbart.py
```

### Author
Raj Nath Patel (patelrajnath@gmail.com)

Linkedin: https://ie.linkedin.com/in/raj-nath-patel-2262b024

### Version
0.1

### LICENSE
Copyright Raj Nath Patel 2020 - present

Pytorch-dl is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public 
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any 
later version.

You should have received a copy of the GNU General Public License along with Pytorch-dl project. 
If not, see http://www.gnu.org/licenses/.
