import torch
import tqdm
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from dataset.data_loader import BertDataSet
from dataset.vocab import WordVocab
from models.bert import Bert
from models.bert_lm import BertLanguageModel

with open("experiments/sample-data/bert-example.txt") as f:
    vocab = WordVocab(f)
    vocab.save_vocab("experiments/sample-data/vocab.pkl")

vocab = WordVocab.load_vocab("experiments/sample-data/vocab.pkl")
data_set = BertDataSet("experiments/sample-data/bert-example.txt", vocab, max_size=512)

lr_warmup = 1000
batch_size = 4

data_loader = DataLoader(data_set, batch_size=4)

vocab_size = len(vocab.stoi)
bert = Bert(vocab_size)
model = BertLanguageModel(bert, vocab_size)

criterion = nn.NLLLoss(ignore_index=0)
optimizer = Adam(lr=0.0001, params=model.parameters())
lr_schedular = lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (lr_warmup / batch_size), 1.0))

if torch.cuda.device_count() > 1:
    print("Using %d GPUS for BERT" % torch.cuda.device_count())
    model = nn.DataParallel(model, device_ids=[0,1,2,3])


for _ in range(10):
    avg_loss = 0
    # Setting the tqdm progress bar
    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc="Running...",
                          total=len(data_loader))
    for i, data in data_iter:
        bert_input, bert_label, segment_label, is_next = data
        bert_out, sentence_pred = model(data[bert_input], data[segment_label])
        mask_loss = criterion(bert_out.transpose(1, 2), data[bert_label])
        next_loss = criterion(sentence_pred, data[is_next])
        loss = next_loss + mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schedular.step()
        avg_loss += loss.item()
    print(avg_loss)