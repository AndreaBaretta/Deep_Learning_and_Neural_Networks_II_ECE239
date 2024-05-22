"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig


MODEL = "bigram"  # bigram or minigpt

if MODEL == "bigram":
    config = BigramConfig
    model = BigramLanguageModel(config)
elif MODEL == "minigpt":
    config = MiniGPTConfig
    model = MiniGPT(config)
else:
    raise ValueError("Invalid model name")


# Initialize wandb if you want to use it
if config.to_log:
    run = wandb.init(project="dl2_proj3")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


train_dataset = TinyStoriesDataset(
    config.path_to_data,
    mode="train",
    context_length=config.context_length,
)
eval_dataset = TinyStoriesDataset(
    config.path_to_data, mode="test", context_length=config.context_length
)

train_dataloader = DataLoader(
    train_dataset, batch_size=config.batch_size, pin_memory=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=config.batch_size, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))


if not Path.exists(config.save_path):
    Path.mkdir(MiniGPTConfig.save_path, parents=True, exist_ok=True)


### ==================== START OF YOUR CODE ==================== ###
"""
You are required to implement the training loop for the model.

Please keep the following in mind:
- You will need to define an appropriate loss function for the model.
- You will need to define an optimizer for the model.
- You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
- It is recommended that you save the model weights every `config.save_iterations` iterations you can also just save the model with the best training loss.

Please check the config file to see the different configurations you can set for the model.
NOTE : 
The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
not a required part of the assignment. 
Feel free to experiment with the parameters and I would be happy to talk to you about them if interested :)
"""

def train_model(model):
    # print(f"{len(train_dataloader)=}")
    # print(f"{len(eval_dataloader)=}")
    # loss_fn = nn.CrossEntropyLoss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
    model.to(device)
    iters = 0
    ret = False
    while True:
        for X_train, y_train in train_dataloader:
            # print("Ayo")
            val_loss = torch.tensor(0)
            X_val, y_val = next(iter(eval_dataloader))
            X_val, y_val = X_val.to(device), y_val.to(device)
            with torch.no_grad():
                # loss = crossentropy(outputs.view(-1,config.vocab_size), targets.view(-1))
                # inputs: (batch_size, num_tokens)

                # targets: (batch_size, num_tokens)

                # outputs: (batch_size, num_tokens, vocab_size)

                val_outputs = model(X_val).squeeze()
                # val_one_hot = nn.functional.one_hot(y_val, config.vocab_size).squeeze().float()
                # print(f'{val_outputs.shape=}')
                # print(f'{val_one_hot.shape=}')
                # print(f'{y_val.shape=}')
                val_loss = nn.functional.cross_entropy(val_outputs.view(-1,config.vocab_size), y_val.view(-1))
                # print(f"{val_outputs.shape=}")
                # print(f"{val_one_hot.shape=}")
                # val_loss = loss_fn(val_outputs, val_one_hot, reduce='sum')
                pass

            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()

            model.train()
            outputs = model(X_train).squeeze()

            # one_hot = nn.functional.one_hot(y_train, config.vocab_size).squeeze().float()
            # print({f"{outputs.shape=}"})
            # print({f"{y_train.shape=}"})

            loss = nn.functional.cross_entropy(outputs.view(-1,config.vocab_size), y_train.view(-1))
            # loss = nn.functional.cross_entropy(outputs, one_hot)
            # loss = loss_fn(outputs, one_hot, reduction='sum')
            loss.backward()
            if config.to_clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                pass
            optimizer.step()
            if iters % (config.max_iter//10) == 0:
                print(f"Training iterations: {iters}/{config.max_iter} - loss={loss.item()}, val_loss={val_loss.item()}")
                pass
            if iters % config.save_iterations == 0:
                # torch.save(model.state_dict(), f"{MODEL}/bigram-{iters}.pt")
                pass                
            run.log({"train-loss": loss.item(), "val-loss": val_loss.item()}, step=iters)
            if iters > config.max_iter:
                ret = True
                break
            iters += 1
            pass
        if ret:
            break
        pass

    run.finish()
    return run