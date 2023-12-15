import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import dill
import numpy as np


df = pd.read_csv('./arxiv100.csv')

le = LabelEncoder()
df['label']= le.fit_transform(df['label'])

labels = df.label
texts = df.title + df.abstract
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.1, random_state=1)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


train_encoding = []
for text in train_texts:
    tokens = tokenizer.encode(text, truncation=True, padding=True, max_length=512)
    train_encoding.append(tokens)

test_encoding = []
for text in test_texts:
    tokens = tokenizer.encode(text, truncation=True, padding=True, max_length=512)
    test_encoding.append(tokens)


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        super().__init__()
        self.data = data_tensor
        self.target = target_tensor

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


sequence_list_tensors = [torch.tensor(seq) for seq in train_encoding]
train_encoding = pad_sequence(sequence_list_tensors, batch_first=True)
sequence_list_tensors = [torch.tensor(seq) for seq in test_encoding]
test_encoding = pad_sequence(sequence_list_tensors, batch_first=True)

train_labels_tensor = torch.zeros(len(train_labels), 10)
for i, label in enumerate(train_labels):
    train_labels_tensor[i, label] = 1

test_labels_tensor = torch.zeros(len(test_labels), 10)
for i, label in enumerate(test_labels):
    test_labels_tensor[i, label] = 1

train_dataset = TensorDataset(train_encoding, train_labels_tensor)
test_dataset = TensorDataset(test_encoding, test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

print(train_loader)
print('Created `train_dataloader` with %d batches!'%len(train_loader))

with open('./arxiv/arxiv_train_dataset.pkl','wb') as f:
    dill.dump(train_dataset, f)

with open('./arxiv/arxiv_train_loader.pkl','wb') as f:
    dill.dump(train_loader, f)
##
with open('./arxiv/arxiv_test_dataset.pkl','wb') as f:
    dill.dump(test_dataset, f)

with open('./arxiv/arxiv_test_loader.pkl','wb') as f:
    dill.dump(test_dataloader, f)

print("dataset done")

