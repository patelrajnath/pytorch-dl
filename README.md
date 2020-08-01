## Pytorch-dl (Deep Learning with Pytorch)
This project implements the classification task using Transformer model. On IMDB sentiment analysis task it achieved a score of 85+ accuracy.

It also contains BERT training- 
* Transformer based Neural MT training and decoding
* Training and fine tuning mBart for Neural MT ([mBart](https://arxiv.org/pdf/2001.08210.pdf))
* Bert encoder ([Default Bert](https://arxiv.org/pdf/1810.04805.pdf))

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

### mBART training
##### Prepare data
```bash
cd examples/translation/
bash prepare-iwslt14.sh

# This will add language tag at the end of each segment in the corpu
sed -e 's/$/ <EN>/' train.en > train-mbart.txt
sed -e 's/$/ <DE>/' train.de >> train-mbart.txt

sed -e 's/$/ <EN>/' valid.en > valid-mbart.txt
sed -e 's/$/ <DE>/' valid.de >> valid-mbart.txt
cd -
bash prep_mbart.sh
```

##### Train model
```bash
bash train_mbart.sh
```
**Note**: This model now could be directly used for NMT training as 
described in the above section. Simply provide the model path (--save_model) and it will 
be automatically used for further fine-tuning. 

**Also, its important to note that we must use the same vocab for corpus preparation, 
the one used for mbart training.** 
Check the sample shell scripts in the following section for both 
corpus preparation and training. 

##### Finetune NMT model
```bash
bash prep_finetune_mbart_nmt.sh
bash finetune_mbart_nmt.sh
```

### IMDB classification:
```bash
python3 classify.py
```

### Bert training:
```bash
python3 pretrain_bert.py
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
