import torch
import datetime
from transformers import GPT2ForSequenceClassification, GPT2Config, get_linear_schedule_with_warmup, AdamW
from transformers import GPT2Tokenizer
import os
# from gpu import (
#     add_gpu_params, 
#     parse_gpu, 
#     distributed_opt, 
#     distributed_gather, 
#     distributed_sync, 
#     cleanup
# )
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object 
from tqdm import tqdm


current_dir = os.getcwd()
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_file_name = os.path.splitext(os.path.basename(__file__))[0] # without the extension
# files dir
checkpoint_path = '/data/clhuang/5212/RCFDA/gpt2_finetune/ckpts/arxiv_2023-12-10_23-29-03.ckpt' # this is for loading the checkpoints
hf_models_cache_dir = '/data/clhuang/huggingface_cache/models'
hf_tokenizer_cache_dir='/data/clhuang/huggingface_cache/tokenizers'
tensorboard_dir = f'{current_dir}/runs/gpt2_lora_{current_time}/'
new_model_checkpoint_dir = f'{current_dir}/ckpts/gpt2_lora' # this is for saving new checkpoints trained in this file

# terminal outputs file:
outputs_file_content = ""
outputs_file_name = f'{current_dir}/terminal_outputs/{current_file_name}__{current_time}.txt'

# hyperparameters
num_epochs = 10
batch_size = 16
max_length = 512



# helper functions
def print_and_store(current_file_content):
	global outputs_file_content 
	outputs_file_content += ('\n' + str(current_file_content))
	print(current_file_content)



# taining step in the github
# def train_validate(
#     model, 
#     optimizer, 
#     scheduler, 
#     train_loader, 
#     valid_loader, 
#     args, 
#     train_step=0, 
#     epoch=0
# ):
#     model.train()
#     avg_lm_loss = AverageMeter()
#     print('start to train the model................', epoch)
#     log_start_time = time.time()
#     best_val_ppl = None

#     train_loader.sampler.set_epoch(epoch)

#     for idx, data in enumerate(train_loader):
#         data = {key: value for key, value in data.items()}

#         _input = data['input'].to(args.device)
#         _target = data['target'].to(args.device)
#         _msk = data['mask'].to(args.device)

#         _lm_logits, _lm_loss = model(
#             _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
#         ) 

#         _lm_loss = _lm_loss.mean() 

#         train_step += 1
#         is_update = True if train_step % args.grad_acc == 0 else False
#         avg_lm_loss.update(_lm_loss.item())
#         optimizer_step(
#             _lm_loss/(args.grad_acc), optimizer, model, scheduler, args, is_update=is_update
#         )
        
#         if train_step % args.log_interval == 0: 
#             elapsed = time.time() - log_start_time
#             lr = optimizer.param_groups[0]['lr']
#             log_str = f'| epoch {epoch:3d} step {train_step:>8d} | { idx + 1:>6d} batches | ' \
#                       f'lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | ' \
#                       f'loss {avg_lm_loss.val:5.2f} | avg loss {avg_lm_loss.avg:5.2f} | ' \
#                       f'ppl {math.exp(avg_lm_loss.avg):5.2f}'

#             if args.rank == 0: 
#                 print(log_str)
#             log_start_time = time.time()
#             avg_lm_loss.reset()
        
#         if train_step % args.save_interval == 0: 
#             if args.rank == 0:
#                 model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
#                 print('saving checkpoint', model_path)
#                 torch.save({'model_state_dict': lora.lora_state_dict(model)}, model_path)
#             distributed_sync(args)

#         # evaluation interval
#         if train_step % args.eval_interval == 0:
#             eval_start_time = time.time()

#             valid_loss, valid_ppl = evaluate(model, valid_loader, args)

#             if best_val_ppl is None or valid_ppl < best_val_ppl:
#                 best_val_ppl = valid_ppl
                
#             log_str = f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | ' \
#                       f'time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:5.2f} | ' \
#                       f'valid ppl {valid_ppl:5.2f} | best ppl {best_val_ppl:5.2f} '

#             if args.rank == 0:
#                 print('-' * 100)
#                 print(log_str)
#                 print('-' * 100)

#             model.train()
#             distributed_sync(args)

#         if train_step == args.max_step:
#             break

#     if args.rank == 0:
#         model_path = os.path.join(args.work_dir, f'model.{train_step}.pt')
#         print('saving checkpoint', model_path)
#         torch.save({'model_state_dict': model.state_dict()}, model_path) 
#     distributed_sync(args)
#     return train_step

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params






if __name__ == '__main__':
    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_and_store('the device is: ' + str(device))
    # end

    # define the model:
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=2, cache_dir=hf_models_cache_dir)
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", config = model_config, cache_dir=hf_models_cache_dir)
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    # load the model
    checkpoint = torch.load(checkpoint_path)
    checkpoint = {k: v for k, v in checkpoint.items() if 'score' not in k}
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint, strict=False)


    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)
   
    total_params, trainable_params = count_parameters(model)
    trainable_percentage = trainable_params / total_params * 100
    print_and_store(f"trainable params BEFORE setting classifier to be trainable: {trainable_params} || all params: {total_params} || trainable%: {trainable_percentage:.10f}")

     # need to set the classifier layer to trainable
    for param in model.score.parameters():
        param.requires_grad = True

    total_params, trainable_params = count_parameters(model)
    trainable_percentage = trainable_params / total_params * 100
    print_and_store(f"trainable params AFTER setting classifier to be trainable: {trainable_params} || all params: {total_params} || trainable%: {trainable_percentage:.10f}")

    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=hf_models_cache_dir)
    tokenizer.pad_token = tokenizer.eos_token


    # optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # datasets
    dataset = load_dataset("imdb")
    print(dataset)
    imdb_test = dataset['test']
    imdb_train = dataset['train']

    print(imdb_train)

    # dataloader
    imdb_train_encodings = tokenizer(list(imdb_train['text']), truncation=True, padding=True,max_length=max_length)
    imdb_train_labels = torch.tensor(list(imdb_train['label']))
    imdb_train_dataset = TensorDataset(torch.tensor(imdb_train_encodings['input_ids']),
                                    torch.tensor(imdb_train_encodings['attention_mask']),
                                    imdb_train_labels)
    imdb_train_loader = DataLoader(imdb_train_dataset, batch_size=batch_size, shuffle=True)

    imdb_test_encodings = tokenizer(list(imdb_test['text']), truncation=True, padding=True,max_length=max_length)
    imdb_test_labels = torch.tensor(list(imdb_test['label']))
    imdb_test_dataset = TensorDataset(torch.tensor(imdb_test_encodings['input_ids']),
                                    torch.tensor(imdb_test_encodings['attention_mask']),
                                    imdb_test_labels)
    imdb_test_loader = DataLoader(imdb_test_dataset, batch_size=batch_size, shuffle=False)

    # tensorboard
    writer = SummaryWriter(log_dir=tensorboard_dir)


    # training
    # print_and_store('======== TRAINING ====================')
    step = 0 # for tensorboard
    for epoch in range(num_epochs):
        print_and_store(f'======== epoch{epoch+1} ====================')
        model.train()
        train_loss = 0
        train_correct = 0
        for batch in tqdm(imdb_train_loader, total=len(imdb_train_loader)):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == labels).float().sum().item()

            # tensorboard
            writer.add_scalar('Loss/train', loss, step)
            step += 1

            # optimizer
            loss.backward()
            optimizer.step()

        train_loss /= len(imdb_train_loader)
        train_accuracy = train_correct / len(imdb_train)
        print_and_store(f"epoch{epoch+1}  training loss = {train_loss}")
        print_and_store(f"epoch{epoch+1}  training accuracy = {train_accuracy}")
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        # testing
        model.eval()
        # print(1)
        with torch.no_grad():
            # print(2)
            test_correct = 0
            for batch in tqdm(imdb_test_loader, total=len(imdb_test_loader)):
                input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
                # print(3)
                outputs = model(input_ids, attention_mask=attention_mask)
                # print(4)
                logits = outputs.logits
                test_correct += (logits.argmax(dim=1) == labels).float().sum().item()
                # print(5)
            test_accuracy = test_correct / len(imdb_test)
            print_and_store(f"epoch{epoch+1}  testing accuracy = {test_accuracy}")
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        #end

        # print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f}, test_accuracy={test_accuracy:.4f}")


    torch.save(model, f"{new_model_checkpoint_dir}/{current_file_name}_{current_time}.ckpt")
    with open(outputs_file_name, 'w') as file:
        file.write(outputs_file_content)