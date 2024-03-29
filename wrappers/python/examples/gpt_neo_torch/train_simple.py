import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTNeoConfig, GPT

os.environ['KMP_DUPLICATE_LIB_OK']='True'

out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 4 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64
window_size = 8
# model
n_layer = 12
attention_types = [["global", "local"], 6]
n_head = 1
n_embd = 24
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile =  False # use PyTorch 2.0 to compile the model to be faster

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = 100
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']
#     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, 
                  block_size=block_size, window_size=window_size, bias=bias, 
                  vocab_size=None, dropout=dropout) # start with model_args from command line
# init a new model from scratch
print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

gptconf = GPTNeoConfig(**model_args)
model = GPT(gptconf)
seq_len = 8
X = torch.randint(0, 20, (batch_size, seq_len))
Y = torch.randint(0, 20, (batch_size * seq_len,))

logits, loss = model(X, Y)

print("n heads = {}".format(n_head))
for p_torch in model.parameters():
    print(p_torch.shape)
        # p_nntile_grad_np = np.zeros(p_nntile.grad.shape, order="F", dtype=np.float32)
        # p_nntile.grad.to_array(p_nntile_grad_np)
        # if p_torch.grad.shape[0] != p_nntile_grad_np.shape[0]:
        #     p_nntile_grad_np = np.transpose(p_nntile_grad_np)

print(loss)