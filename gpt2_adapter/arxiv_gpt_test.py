# -*- coding: utf-8 -*-
import os
import torch
from transformers import GPT2Tokenizer,GPT2Config, AdamW, GPT2ForSequenceClassification
from tqdm import tqdm
import adapters
from adapters import init, BnConfig
import dill

hf_models_cache_dir = './model_cache'
outputs_file_content = ""
batch_size = 16

def print_and_store(current_file_content):
    global outputs_file_content
    outputs_file_content += ('\n' + str(current_file_content))
    print(current_file_content)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

with open('./arxiv/arxiv_test_loader.pkl', 'rb') as f:
    test_loader = dill.load(f)

#######################################
# model, tokenize
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=hf_models_cache_dir)
tokenizer.pad_token = tokenizer.eos_token
config = GPT2Config.from_pretrained("gpt2",)
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=10, cache_dir=hf_models_cache_dir).to(device)


task_name = "IMDB"
adapters.init(model)
config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=96, non_linearity="relu")
model.add_adapter(task_name)
model.train_adapter(task_name)
model.set_active_adapters(task_name)
model.config.pad_token_id = model.config.eos_token_id

checkpoint_path = "./output/fted_gpt2.ckpt"
checkpoint = torch.load(checkpoint_path)
checkpoint = {k: v for k, v in checkpoint.items() if 'score' not in k}
model.load_state_dict(checkpoint, strict=False)

checkpoint_path = "./arxiv_2023-12-10_23-29-03.ckpt"
checkpoint = torch.load(checkpoint_path)
checkpoint = {k: v for k, v in checkpoint.items() if 'score' in k}
model.load_state_dict(checkpoint, strict=False)

model.to(device)
#######################################


# testing
with tqdm(total=len(test_loader), desc='testing') as pbar:
    with torch.no_grad():
        # for batch in tqdm(test_loader, total=len(test_loader)):
        acc = 0
        for batch in test_loader:
            inputs = {'input_ids': batch[0].to(device), 'labels': batch[1].to(device)}
            outputs = model(**inputs)
            logits = outputs.logits
            max_indices = logits.argmax(dim=1)
            for i in range(len(batch[1])):
                ind = int(max_indices[i])
                if batch[1][i][ind] == 1:
                    acc = acc + 1
            # print(f'batch is {batch}')
            pbar.update(1)
        accuracy = acc / (len(test_loader) * batch_size)
        print_and_store(f"Testing accuracy = {accuracy}")

with open("./output/output_arxiv.txt", 'w') as file:
    file.write(outputs_file_content)
