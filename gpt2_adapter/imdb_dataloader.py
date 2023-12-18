from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import dill
import torch
from transformers import GPT2Tokenizer

dataset = load_dataset("imdb")
print(dataset)
imdb_test = dataset['test']
imdb_train = dataset['train']

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


# Test on IMDB dataset
imdb_test_encodings = tokenizer(list(imdb_test['text']), truncation=True, padding=True, max_length=512)
imdb_test_labels = torch.tensor(list(imdb_test['label']))
imdb_test_dataset = TensorDataset(torch.tensor(imdb_test_encodings['input_ids']),
                                  torch.tensor(imdb_test_encodings['attention_mask']),
                                  imdb_test_labels)
imdb_test_loader = DataLoader(imdb_test_dataset, batch_size=1, shuffle=False)

# Train on IMDB dataset
imdb_train_encodings = tokenizer(list(imdb_train['text']), truncation=True, padding=True, max_length=512)
imdb_train_labels = torch.tensor(list(imdb_train['label']))
imdb_train_dataset = TensorDataset(torch.tensor(imdb_train_encodings['input_ids']),
                                   torch.tensor(imdb_train_encodings['attention_mask']),
                                   imdb_train_labels)
imdb_train_loader = DataLoader(imdb_train_dataset, batch_size=4, shuffle=True)

type(imdb_test_loader)

with open('./dataset/imdb_train_dataset.pkl','wb') as f:
    dill.dump(imdb_train_dataset, f)

with open('./dataset/imdb_train_loader.pkl','wb') as f:
    dill.dump(imdb_train_loader, f)
##
with open('./dataset/imdb_test_dataset.pkl','wb') as f:
    dill.dump(imdb_test_dataset, f)

with open('./dataset/imdb_test_loader.pkl','wb') as f:
    dill.dump(imdb_test_loader, f)

print("dataset done")
