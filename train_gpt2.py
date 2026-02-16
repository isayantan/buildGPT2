import math
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projection for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output ptojection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()   # batch size, sequence length, embedding dimensionality

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)   
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        # attention
        # att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim = -1)
        # y = att @ v

        # instead implement flash attention with scaled_dot_product_attention, which is much faster and more memory efficient
        y = F.scaled_dot_product_attention(q, k, v, is_causal = True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate = 'tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """Configuration for a GPT-style language model."""

    vocab_size: int = 50257                # no of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    block_size: int = 1024                 # context length
    n_layer: int = 12                    
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # check "Language Models are Unsupervised Multitask Learners" paper for more details on the 0.02 value and scaling factor
            # for the linear layers in the attention and MLP, we use a scaling 1/\sqrt{number of layers} for initialization to improve stability
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets = None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        # forward the token and position embedding
        pos = torch.arange(0, T, dtype = torch.long, device = idx.device) # shape T
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Loads pretrained GPT-2 weights from huggingface
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel
        print("loading weights from the pretrained gpt: %s" % model_type)

        # n_layer, n_head, n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer = 12, n_head = 12, n_embd = 768),  # 124M params
            'gpt2-medium': dict(n_layer = 24, n_head = 16, n_embd = 1024), # 350M
            'gpt2-large': dict(n_layer = 36, n_head = 20, n_embd = 11280), # 774M
            'gpt2-xl': dict(n_layer = 40, n_head = 25, n_embd = 1600),
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT2 checkpoints
        config_args['block_size'] = 1024  # always 1024 context length for GPT2 checkpoints
        # create a from-scratch GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer
        
        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]      # discard buffer masked bias
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]             # discard buffer bias
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched key: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transpose the Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # copy the other weights
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model 
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # collect parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        # assign the parameters to the optimizer groups, using weight decay for the appropriate parameters
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # create AdamW optimizer and use the fused version if available for faster training
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = use_fused)
        return optimizer

# -----------------------------------------------------------------
# data loading
import tiktoken
file = "data/tiny_shakespeare.txt"

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B 
        self.T = T

        # at init load tokens from the file
        with open(file, 'r') as f:
            text = f.read()
        # encode with tiktoken gpt2 encoder
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        # current pos
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        # advance the pos in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bound then reset current_position
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# -----------------------------------------------------------------
import time

# set device to cuda
device = "cuda:2"

# set seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

train_loader = DataLoaderLite(B = 16, T = 1024)

# set float32 matmul precision to high for faster training on Ampere and newer GPUs (like the 4090) - this is a no-cost speedup
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

# learning rate schedule with warmup and cosine decay
max_lr = 6e-4
min_lr = max_lr * 0.1
warmpup_steps = 10
max_steps = 50

def get_lr(it):
    # linear warmup
    if it < warmpup_steps:
        return max_lr * (it + 1)/ warmpup_steps
    if it > max_steps:
        return min_lr
    # cosine decay
    decay_ratio = (it - warmpup_steps)/ (max_steps - warmpup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimizer!
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4, betas = (0.9, 0.95), eps = 1e-8)
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device = device)

for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # use mixed precision with autocast. Ampere and newer GPUs have hardware support for bfloat16, which is
    #  a 16-bit floating point format that has the same range as float32 but with less precision.
    #  This allows us to use mixed precision training without worrying about underflow or overflow issues that can arise with float16.
    #  By using autocast, we can automatically use bfloat16 where it's beneficial and keep other operations in float32 for better accuracy.
    with torch.autocast(device_type = device, dtype = torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    # clip the gradient to 1.0 to prevent exploding gradients, which can destabilize training
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000          # convert to milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T)/(t1 - t0)         # throghput in tokens per second
    print(f"step {step} | loss: {loss.item()} | norm: {norm.item():.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

import sys; sys.exit(0)

# num_return_sequences = 5
# max_length = 30
# device = 'cuda:2'

# # model = GPT.from_pretrained('gpt2')
# model = GPT(GPTConfig())
# model.eval()
# model.to(device)

# # prefix tokens
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a languange model,")
# tokens = torch.tensor(tokens, dtype = torch.long)  # (8, )
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
# x = tokens.to(device)

# # generate! right now x is (B, T) where B = 5, T = 8
# # set seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#     # forward the model to get the logits
#     with torch.no_grad():
#         logits = model(x)  # (B, T, vocab_size)
#         # get the logit at last position
#         logits = logits[:, -1, :]  # (B, vocab_size)
#         # get prob
#         probs = F.softmax(logits, dim = -1)
#         # do top-k sampling of 50
#         # topk_probs here becomes (5, 50), topk_indices is (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
#         # select a token from the top-k probs
#         ix = torch.multinomial(topk_probs, 1) # (B, 1)
#         # gather the corresponding indices
#         xcol = torch.gather(topk_indices, -1, ix)
#         # append to seq
#         x = torch.cat((x, xcol), dim = 1)

# # print the generated seq
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded)




