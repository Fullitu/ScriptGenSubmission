import inspect
import numpy as np
import pandas as pd
import torch
import math
import os
import torch.nn as nn
import time
import sentencepiece as spm

from helper_functions import load_bpe
from simlpe_tokenizer import CharacterTokenizer
from datetime import datetime
from dataclasses import dataclass
from torch.nn import functional as F
from tokenizers import Tokenizer, processors
from tokenizers.decoders import ByteLevel as ByteLevelDecoder, Metaspace as MetaspaceDecoder, WordPiece as WordPieceDecoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

# Model Parameters
torch.manual_seed(12345)

""" After running:"""

@dataclass
class Config:
    vocab_size: int = None
    n_layer = 8
    n_head = 8
    n_embd = 512
    max_steps = 2751
    warmup_steps = 50
    max_lr = 5e-4
    min_lr = max_lr*0.2
    lr_low = 2000                                   # after which step lr becomes min lr
    weight_decay = 0.1
    eval_interval = 50
    sample_generations = 5
    sample_max_length = 128
    sample_interval = 100
    checkpoint_interval = 250
    dropout: float = 0.4
    B = 64
    T = 1024
    block_size: int  = T

""" Model structure - based on GPT-2 """

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # bias/mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B,T,C = x.size() # unpacking batch, sequence length, and embedding dimensionality(n_embd)
        # calculate query, key, values for all heads in batch and move head forward to the batch
        # nh is number of heads, hs is head size, and C(number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        # attention (materialize the large (T,T) attention matrix for all queries and keys)

        """
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))                  # old attention calculation
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs) "weighted sum" kinda
        """

        # flash attewntion
        y = F.scaled_dot_product_attention(q,k,v, dropout_p=config.dropout, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)           # Fully connected layer


        self.gelu = nn.GELU()                                           # Gaussian Error Linear Unit activation,
                                                                        # like a relu but not exactly
                                                                        # better performance than relu, removes dead
                                                                        # neurons(0 gradient) and vanishing gradient
        self.dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)         # Fully connected layer
        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln_1(x)))         # Attention - communication between tokens, aggregation of information
        x = x + self.dropout(self.mlp(self.ln_2(x)))          # MLP - multi-layer perceptron, tokens think about information gathered
        return x


class ScriptGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),                   # Embedding for input tokens
            wpe = nn.Embedding(config.block_size, config.n_embd),                   # Embedding for positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),      # Stacked transformer layers(hidden layers)
            ln_f = nn.LayerNorm(config.n_embd)                                      # Layer norm for output
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)      # Output layer, final classifier

        # weight sharing scheme (WTE at the bottom and classifier at the top of transformer)
        self.transformer.wte.weight = self.lm_head.weight
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5                # scale standard deviation according to paper (1/sqrt(N dim)), 2 cause transformer has 2 layers inside
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    def forward(self, idx, targets=None):
        # idx with shape B,T
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward, model block size is {self.config.block_size} but got sequence of length {T}"
        # forward token and pos embedding
        pos = torch.arange(0,T, dtype=torch.long, device=idx.device) # shape T
        pos_emb = self.transformer.wpe(pos) # position embedding of shape T,n_embd
        tok_emb = self.transformer.wte(idx) # token embedding of shape B,T,n_embd
        x = tok_emb + pos_emb
        # forward the blocks of the transofrmer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # B,T,vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all candidate params (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any param that is 2D will be weight decayed, otherwise no.
        # for example, all weight tensors in matmuls + embeddings decay, all biases and layernoms do not
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, num decayed parameters: {num_decay_params:,}")
        print(f"num nodecayed parameter tensors:{len(nodecay_params)}, num nondecayed parameters: {num_nodecay_params:,}")
        # Create AdamW optimizer
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"Using {'fused' if use_fused else 'python'} adamw optimizer.")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay, fused=use_fused)
        return optimizer

    def generate(self, tokenizer, completion="", max_tokens=64, n_generations = 5, top_k=10, seed=None):
        model.eval()
        num_generations = n_generations
        max_length = max_tokens
        # for classic bpe changed
        # tokens = tokenizer.encode(completion)
        tokens = tokenizer.encode_to_ids(completion)
        # tokens = tokenizer.encode(completion).ids
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_generations, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(12345)
        while xgen.size(1) < max_length:
            # forward the model to get logits
            with torch.no_grad():
                logits, loss = model(xgen)  # B,T,vocab_size
                # take the logitrs at the last pos
                logits = logits[:, -1, :]  # B,vocab_size
                # get probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of top 50
                # topk_probs becomes (5,50) topk_indices is (5,50)
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                # select token from top-k probs
                # multinomial does not need input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B,1
                # get corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B,1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # decode and print
        text = ""
        for i in range(num_generations):
            text += f"Generation {i + 1}:\n{tokenizer.decode_from_ids(xgen[i].tolist())}\n"
        print(text)
        return text


class CustomDataLoader:
    def __init__(self, B, T, tokenizer, split):
        self.B = B
        self.T = T
        self.dataset = {}
        assert split in {"train", "test"}, "Split must be either 'train' or 'test'"
        # at init load tokens and store in memory
        with open("cleaned_text/combined.txt", "r", encoding="utf-8") as f:
            text = f.read()
            text = text.replace('�', '')
            print(f"Combined text loaded.\n")
        # for classic bpe changed
        tokens = tokenizer.encode_to_ids(text)
        # tokens = tokenizer.encode(text)
        # tokens = tokenizer.encode(text).ids
        split_idx = int(len(tokens) * 0.9)
        self.dataset = {
            "train": torch.tensor(tokens[:split_idx], device = device),
            "test": torch.tensor(tokens[split_idx:], device = device)
        }
        self.tokens = self.dataset[split]
        print(f"Loaded {len(tokens)} tokens.")
        print(f"1 epoch = {len(tokens)//(B*T*grad_accumulation_steps)} batches")
        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in tensor
        self.current_position += B*T
        # if loading next batch overshoots the tokens, reset the position
        if self.current_position +(B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y

def get_lr(it):
    # linear warmup for warmup_iters steps
    if it < config.warmup_steps:
        return config.max_lr * (it+1) / config.warmup_steps
    # if it > lr_decay_iters, return min learning rate
    if it >config.lr_low:
        return config.min_lr
    # in between use cosine decay down to min lr
    decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff 1 -> 0
    return config.min_lr + coeff * (config.max_lr-config.min_lr)

""" Model/checkpoint saving """

def save_checkpoint(model, optimizer, step, logs, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"m_T{config.T}_B{config.B}_nL{config.n_layer}_nE{config.n_embd}_nH{config.n_head}_{datetime.today().strftime('%Y-%m-%d_%H-%M')}_{tok_file}_voc_{tokenizer.get_vocab_size()}_s_{step}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "val_dataloader": val_loader,
        "train_dataloader": train_loader,
        "logs": logs
    }, checkpoint_path)
    print(f"Model saved at step {step} to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path = "checkpoints"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    val_loader = checkpoint["val_dataloader"]
    train_loader = checkpoint["train_dataloader"]
    step = checkpoint["step"]
    print(f"Model loaded from step {step}")
    return model, optimizer, step, val_loader, train_loader

def log_data(dataframe, step, loss, val_loss):
    new_row = pd.DataFrame([{"step": step, "loss": loss, "val_loss": val_loss}])
    dataframe = pd.concat([dataframe, new_row], ignore_index=True)
    return dataframe

def save_logs(logs):
    logs.to_csv(f"analysis/log_{datetime.today().strftime('%Y-%m-%d_%H-%M')}_{tok_file}_voc_{tokenizer.get_vocab_size()}_s_{step}.csv", index=False)
    print("Logs saved.")
""" Tokenizer and Data loading, compiling model"""


# Init tokenizer

tok_file = "bpe_classic.json"
tokenizer = load_bpe(tok_file)  # path to your saved tokenizer

# Character tokenizer
# tok_file = "character_tokenizer_1.json"
# tokenizer = CharacterTokenizer(token_length=1)
# print("Tok keys",tokenizer.itos.keys())

# #BPE
# tok_file = "bpe_tokenizer.json"
# tokenizer = Tokenizer.from_file(tok_file)
# tokenizer.decoder = ByteLevelDecoder()

# # UNIGRAM
# tok_file = "Unigram_tokenizer.json"
# tokenizer = Tokenizer.from_file(tok_file)
# tokenizer.decoder = MetaspaceDecoder()

# WORDPIECE
# tok_file = "WordPiece_tokenizer.json"
# tokenizer = Tokenizer.from_file(tok_file)
# tokenizer.decoder = WordPieceDecoder()


# tokenizer.post_processor = processors.BertProcessing(
#     ("[SEP]", tokenizer.token_to_id("[SEP]")),
#     ("[CLS]", tokenizer.token_to_id("[CLS]")),
# )

tok_file = tok_file.replace(".json", "")
vocab_size = tokenizer.get_vocab_size()
config = Config(vocab_size=vocab_size)

total_batch_size = 524288           #131072 0.5M roughly, same as small variant of gpt3, since im using gpt3-small hyperparams
B = config.B                        # mini batch size
T = config.T                        # sequence length

assert total_batch_size % (B*T) == 0, "Batch size must be divisible by mini batch dimensions"
grad_accumulation_steps = total_batch_size // (B*T)
print(f"Total batch size: {total_batch_size:,}, B: {B}, T: {T}, grad accumulation steps: {grad_accumulation_steps}")

train_loader = CustomDataLoader(B,T, tokenizer, "train")
val_loader = CustomDataLoader(B,T, tokenizer, "test")

torch.set_float32_matmul_precision('high')               # Setting TF32 precision, speed up training
model = ScriptGenerator(config).to(device)

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model)
    print("Model compiled.\n\n")
except RuntimeError as e:
    print("Triton not available, model not compiled, proceeding without compilation")

""" Execution"""
# training
torch.manual_seed(12345)
print(f"Device type:{device.type}")
optimizer = model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=config.max_lr, device=device.type)

print("Train/Generate? (t/g)")
if input() != "t":
    data = ""
    print("Enter weights file path:")
    model, optimizer, step, val_loader, train_loader = load_checkpoint(model, optimizer, checkpoint_path=input())
    #print("Enter prompt(or leave empty):")
    #data += model.generate(tokenizer, completion=f"", max_tokens=config.T//4, top_k=10, n_generations=5)
    data += model.generate(tokenizer, completion=f"SCENE ONE \n", max_tokens=config.T//4, top_k=10, n_generations=5)
    data += model.generate(tokenizer, completion=f"SCENE ONE\n\nKRAMER\n\n", max_tokens=config.T//4, top_k=10, n_generations=5)
    with open(f"generations\\{tok_file}.txt", "w") as f:
        f.write(data)
    exit(0)


start_step = 0
print("Load checkpoint? (y/n)")
if input() == "y":
    print("Enter checkpoint path(from root):")
    model, optimizer, start_step, val_loader, train_loader = load_checkpoint(model, optimizer, input())
    print("Checkpoint loaded.")
    print(f"Resuming training from step {start_step}")


print("Starting training...")

logs = pd.DataFrame(columns=["step", "loss", "val_loss"])
val_loss_accum = None

for step in range(start_step, config.max_steps+1):
    t0 = time.time()

    """ Eval """
    if step % config.eval_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 100
            for _ in range(val_loss_steps):
                x,y = val_loader.next_batch()
                x,y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits, loss = model(x,y)
                loss = loss/val_loss_steps
                val_loss_accum += loss.detach()
        print(f"[!] Eval [!] step: {step}, val_loss: {val_loss_accum.item():.6f}")

    """ Sample generations """
    if step >0 and step%config.sample_interval == 0:
        model.eval()
        num_generations = config.sample_generations
        max_length = config.sample_max_length
        # for classic bpe changed
        # tokens = tokenizer.encode("SCENE ONE").ids
        tokens = tokenizer.encode_to_ids("SCENE ONE")
        # tokens = tokenizer.encode("SCENE ONE")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_generations, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(12345+step)
        while xgen.size(1)<max_length:
            # forward the model to get logits
            with torch.no_grad():
                logits, loss = model(xgen)  # B,T,vocab_size
                # take the logitrs at the last pos
                logits = logits[:,-1,:] # B,vocab_size
                # get probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of top 50
                # topk_probs becomes (5,k) topk_indices is (5,k)
                topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)
                # select token from top-k probs
                # multinomial does not need input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B,1
                # get corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B,1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # decode and print
        for i in range(num_generations):
            print(f"Sample {i+1}: {tokenizer.decode_from_ids(xgen[i].tolist())}")

    """ Train """
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0# sets gradients to 0
    for microstep in range(grad_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type = device.type, dtype = torch.bfloat16): # mixed precision training, speed up training (BFloat16) https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            logits, loss = model(x,y)                               # grabs loss
        """
        Has to be done because the gradients just add on all backward calls
        adding gradients is sum in objective, we want mean. Scale loss by grad_accumulation_steps
        """
        loss = loss/grad_accumulation_steps                         # scale loss
        loss_accum += loss.detach()
        loss.backward()                                             # backpropagation, adds gradients to parameters

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping, prevents exploding gradients

    # get learning rate for this iteration
    lr = get_lr(step)                                           # learning rate scheduler
    for param_group in optimizer.param_groups:                  # update learning rate, we need to iterate for some reason? only one param group either way
        param_group['lr'] = lr

    optimizer.step()                                            # update parameters
    torch.cuda.synchronize()                                    # waiting for gpu to sync, remove for training
    t1 = time.time()
    dt = (t1-t0)*1000                                           # time diff in ms
    tokens_processed = train_loader.B * train_loader.T * grad_accumulation_steps
    tokens_per_sec = tokens_processed/(t1-t0)    # tokens per second
    print(f"step: {step}, loss: {loss_accum.item():.6f}, norm: {norm:.4f}, lr: {lr:.4e}, dt:{dt:.2f}ms, tok/s: {tokens_per_sec:.2f}")


    if step%config.eval_interval == 0:
        logs = log_data(logs, step, loss_accum.item(), val_loss_accum.item())

    if step % config.checkpoint_interval == 0 and step >0 and step != start_step:
        save_checkpoint(model, optimizer, step, logs, "checkpoints")
        save_logs(logs)

print("Training complete. Saving model and logs...")
save_logs(logs)
save_checkpoint(model, optimizer, config.max_steps, logs, "weights")
