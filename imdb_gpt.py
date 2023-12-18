# -*- coding: utf-8 -*-
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer,GPT2Config, AdamW, GPT2ForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from tqdm import tqdm
import dill
import adapters
from adapters import init, BnConfig


hf_models_cache_dir = '/data/csongak/huggingface_cache/models/'
outputs_file_content = ""


def print_and_store(current_file_content):
    global outputs_file_content
    outputs_file_content += ('\n' + str(current_file_content))
    print(current_file_content)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

#######################################
# dataset preprocess
dataset = load_dataset("imdb", cache_dir=hf_models_cache_dir)
imdb_test = dataset['test']
imdb_train = dataset['train']

# load
with open('./dataset/imdb_train_dataset.pkl', 'rb') as f:
    imdb_train_dataset = dill.load(f)

with open('./dataset/imdb_train_loader.pkl', 'rb') as f:
    imdb_train_loader = dill.load(f)
##
with open('./dataset/imdb_test_dataset.pkl', 'rb') as f:
    imdb_test_dataset = dill.load(f)

with open('./dataset/imdb_test_loader.pkl', 'rb') as f:
    imdb_test_loader = dill.load(f)
#######################################


#######################################
# model, tokenize
checkpoint_path = "./arxiv_2023-12-10_23-29-03.ckpt"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=hf_models_cache_dir)
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained("gpt2",)
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=2, cache_dir=hf_models_cache_dir).to(device)
checkpoint = torch.load(checkpoint_path)
checkpoint = {k: v for k, v in checkpoint.items() if 'score' not in k}
model.load_state_dict(checkpoint, strict=False)

task_name = "IMDB"
adapters.init(model)
config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=96, non_linearity="relu")
model.add_adapter(task_name)
model.train_adapter(task_name)
model.set_active_adapters(task_name)
model.config.pad_token_id = model.config.eos_token_id
model.to(device)
#######################################
"""
#######################################
model.eval()
with torch.no_grad():
    imdb_correct = 0
    with tqdm(total=len(imdb_test_loader), desc='testing') as pbar:
        for batch in imdb_test_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            imdb_correct += (logits.argmax(dim=1) == labels).float().sum().item()
            pbar.update(1)
        imdb_accuracy = imdb_correct / len(imdb_test)

print('imdb gpt2 vanilla accuracy:', imdb_accuracy)
#######################################

"""
#######################################
#Hyper-parameters: batch_size = 16, lr=1e-5, AdamW, epoch=5
# Train on IMDB dataset
optimizer = AdamW(model.parameters(), lr=1e-5)
epoch_num = 5
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(epoch_num):
    model.train()
    train_loss = 0
    train_correct = 0
    with tqdm(total=len(imdb_train_loader), desc='testing') as pbar:
        for batch in imdb_train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == labels).float().sum().item()
            loss.backward()
            optimizer.step()
            pbar.update(1)
    train_loss /= len(imdb_train_loader)
    train_accuracy = train_correct / len(imdb_train)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    with torch.no_grad():
        test_correct = 0
        for batch in imdb_test_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            test_correct += (logits.argmax(dim=1) == labels).float().sum().item()
        test_accuracy = test_correct / len(imdb_test)
        test_accuracies.append(test_accuracy)

    print_and_store(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, test_accuracy={test_accuracy:.4f}")
#######################################
torch.save(model.state_dict(), "./output/fted_gpt2.ckpt")

with open("./output/output.txt", 'w') as file:
    file.write(outputs_file_content)

