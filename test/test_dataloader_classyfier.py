import sys

import torch
import tqdm

from dataset.config import Config
from dataset.dataloader_classifier import Dataset

config = Config()
train_file = 'sample-data/200410_train_stratshuf_english.csv'
if len(sys.argv) > 2:
    train_file = sys.argv[1]
test_file = 'sample-data/200410_test_stratshuf_chinese_200410_english.csv'
if len(sys.argv) > 3:
    test_file = sys.argv[2]

dataset = Dataset(config)
dataset.load_data(train_file, test_file)

NUM_CLS = len(dataset.vocab)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

for batch in tqdm.tqdm(dataset.train_iterator):
    input = batch.text.to(device)
    label = batch.label - 1
    label = label.to(device)
    print("INPUT", input)
    print("LABEL", label)