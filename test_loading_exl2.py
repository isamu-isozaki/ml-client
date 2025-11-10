import asyncio
import json
import os
import time
import configparser
import argparse
from typing import AsyncIterable, List, Generator, Union, Optional
import traceback
import subprocess

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, TextIteratorStreamer
from threading import Thread
import queue
import traceback
import re


import sys, os
import uvicorn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Cache_TP,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

from exllamav2.attn import ExLlamaV2Attention
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.moe_mlp import ExLlamaV2MoEMLP
from exllamav2.parallel_decoder import ExLlamaV2ParallelDecoder

import argparse, os, math, time
import torch
import torch.nn.functional as F
from exllamav2.conversion.tokenize import get_tokens
from exllamav2.conversion.quantize import list_live_tensors
import gc

# from exllamav2.mlp import set_catch

import sys
import json

torch.cuda._lazy_init()
torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 150)

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_float32_matprec("medium")

from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
import uuid
from blessed import Terminal
import textwrap
from outlines.integrations.exllamav2 import RegexFilter, JSONFilter, ChoiceFilter
from util_merge import ExLlamaV2MergePassthrough

def generate_unique_id():
    return uuid.uuid4()

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 100  # default value of 100
    temperature: Optional[float] = 0.0  # default value of 0.0
    stream: Optional[bool] = False  # default value of False
    best_of: Optional[int] = 1
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0  # default value of 0.0
    presence_penalty: Optional[float] = 0.0  # default value of 0.0
    log_probs: Optional[int] = 0  # default value of 0.0
    n: Optional[int] = 1  # default value of 1, batch size
    suffix: Optional[str] = None
    top_p: Optional[float] = 0.0  # default value of 0.0
    user: Optional[str] = None
    outlines_type: Optional[str] = None
    choices: Optional[list[str]] = None
    regex: Optional[str] = None
    json: Optional[str] = None
    request_id: Optional[str] = None

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 100  # default value of 100
    temperature: Optional[float] = 0.0  # default value of 0.0
    stream: Optional[bool] = False  # default value of False
    frequency_penalty: Optional[float] = 0.0  # default value of 0.0
    presence_penalty: Optional[float] = 0.0  # default value of 0.0
    log_probs: Optional[int] = 0  # default value of 0.0
    n: Optional[int] = 1  # default value of 1, batch size
    top_p: Optional[float] = 0.0  # default value of 0.0
    user: Optional[str] = None
    outlines_type: Optional[str] = None
    choices: Optional[list[str]] = None
    regex: Optional[str] = None
    json: Optional[str] = None
    request_id: Optional[str] = None
    partial_generation: Optional[str] = None

#repo_str = 'theprofessor-exl2-speculative'

parser = argparse.ArgumentParser(description='Run server with specified port.')

# Add argument for port with default type as integer
parser.add_argument('--port', type=int, help='Port to run the server on.')
parser.add_argument('--repo_str', type=str, default='llama3-70b-instruct', help='The model repository name')
parser.add_argument('--max_chunk_size', type=int, default=15872, help='Max chunk size.')
parser.add_argument('--max_new_tokens', type=int, default=15872, help='Max new tokens.')
parser.add_argument('--use_draft_model', action="store_true", help='Do speculative decoding')
parser.add_argument('--not_paged', action="store_true", help='Do not do paged attention')

# Add arguments from the new model loading code
parser.add_argument("-pnb", "--prompt_no_bos", action="store_true", help="Don't add BOS token to prompt")
parser.add_argument("-t", "--tokens", type=int, default=15872, help="Max no. tokens")
parser.add_argument("-mix", "--mix_layers", type=str, help="Load replacement layers from secondary model. Example: --mix_layers 1,6-7:/mnt/models/other_model")
parser.add_argument("-nwu", "--no_warmup", action="store_true", help="Skip warmup before testing model")
parser.add_argument("-sl", "--stream_layers", action="store_true", help="Load model layer by layer (perplexity evaluation only)")
parser.add_argument("-sp", "--standard_perplexity", choices=["wiki2"], help="Run standard (HF) perplexity test, stride 512 (experimental)")
parser.add_argument("-rr", "--rank_reduce", type=str, help="Rank-reduction for MLP layers of model, in reverse order (for experimentation)")
parser.add_argument("-mol", "--max_output_len", type=int, help="Set max output chunk size (incompatible with ppl tests)")
model_init.add_args(parser)
# Parse the arguments
args = parser.parse_args()
repo_str = args.repo_str

term = Terminal()

class StatusArea:
    def __init__(self, num_lines):
        self.num_lines = min(num_lines, term.height - 8)  # Ensure we don't exceed terminal height
        self.messages = [""] * num_lines

    def update(self, message, line=None):
        if line is not None:
            # Update a specific line
            if 0 <= line < self.num_lines:
                self.messages[line] = message
        else:
            # Handle multi-line message
            lines = message.split('\n')
            if len(lines) > self.num_lines:
                # Truncate to last num_lines if exceeds num_lines
                lines = lines[-self.num_lines:]
            
            # Update messages, padding with empty strings if necessary
            self.messages = lines + [""] * (self.num_lines - len(lines))

        self.display()

    def display(self):
        for i, message in enumerate(self.messages):
            wrapped_message = textwrap.shorten(message, width=term.width, placeholder="...")
            print(term.move_xy(0, i) + term.clear_eol + wrapped_message)
        
        # Move cursor below the status area
        print(term.move_xy(0, self.num_lines), end='', flush=True)


class JobStatusDisplay:

    def __init__(self, job, status_lines):
        #self.console_line = console_line + status_lines
        self.console_line = None
        self.job = job
        self.prefill = 0
        self.max_prefill = 0
        self.collected_output = ""
        self.tokens = 0
        self.spaces = " " * term.width
        self.status_lines = status_lines
        self.display_text = ""
        #text = term.black(f"{self.console_line:3}:")
        #text += term.blue("enqueued")
        #print(term.move_xy(0, self.console_line) + text)

    def update_position(self, index):
        self.console_line = self.status_lines + index
        self.init_display_text()

    def init_display_text(self):
        self.display_text = term.black(f"{self.console_line:3}:") + term.blue("enqueued")


    def update(self, r):
        if self.console_line is None:
            return  # Skip update if position hasn't been set yet
        stage = r["stage"]
        stage = r.get("eos_reason", stage)

        self.collected_output += r.get("text", "").replace("\n", "\\n")

        token_ids = r.get("token_ids", None)
        if token_ids is not None: self.tokens += token_ids.shape[-1]

        self.prefill = r.get("curr_progress", self.prefill)
        self.max_prefill = r.get("max_progress", self.max_prefill)

        text = term.black(f"{self.console_line:3}:")
        text += term.blue(f"{stage:16}")
        text += "prefill [ " + term.yellow(f"{self.prefill: 5} / {self.max_prefill: 5}")+" ]"
        text += "   "
        text += term.green(f"{self.tokens: 5} t")
        text += term.black(" -> ")
        output_length = term.width - len(text) +20
        text += (self.spaces + self.collected_output)[-output_length:].replace("\t", " ")

        if "accepted_draft_tokens" in r:
            acc = r["accepted_draft_tokens"]
            rej = r["rejected_draft_tokens"]
            eff = acc / (acc + rej) * 100.0
            text += term.bright_magenta(f"   SD eff.: {eff:6.2f}%")

        #print(term.move_xy(0, self.console_line) + text)
        self.display_text = text

    def display(self):
        if self.console_line is not None:
            print(term.move_xy(0, self.console_line) + self.display_text)


config = configparser.ConfigParser()
config.read('config.ini')

repo_id = config.get(repo_str, 'repo')
specrepo_id = config.get(repo_str, 'specrepo')
host = config.get('settings', 'host')
# Max individual context
max_context = int(config.get(repo_str, 'max_context'))
# Total number of tokens to allocate space for. This is not the max_seq_len supported by the model but
# the total to distribute dynamically over however many jobs are active at once
total_context = int(config.get(repo_str, 'total_context'))

config_eos_token_ids = config.get(repo_str, 'eos_token_ids', fallback=None)

port = args.port if args.port is not None else config.getint('settings', 'port')
display_mode = 1

# Whether to use paged mode or not. The generator is very handicapped in unpaged mode, does not support batching
# or CFG, but it will work without flash-attn 2.5.7+
paged = not args.not_paged

# Where to find our model
model_dir = repo_id

# N-gram or draft model speculative decoding. Largely detrimental to performance at higher batch sizes.
use_ngram = False
use_draft_model = args.use_draft_model
if use_draft_model:
    model_dir = repo_id
    draft_model_dir = specrepo_id

# Max number of batches to run at once, assuming the sequences will fit within total_context.
max_batch_size = 4 if paged else 1

# Max chunk size. Determines the size of prefill operations. Can be reduced to reduce pauses whenever a
# new job is started, but at the expense of overall prompt ingestion speed.
max_chunk_size = args.max_chunk_size

# Max new tokens per completion. For this example applies to all jobs.
max_new_tokens = args.max_new_tokens

# Demonstrate token healing
healing = True

# Initialize model and tokenizer using the new approach
print("Initializing model with new approach...")

# Check conflicting settings
if hasattr(args, 'stream_layers') and args.stream_layers:
    if hasattr(args, 'gpu_split') and args.gpu_split:
        print(" ## Can only use one GPU when streaming layers")
        sys.exit()

# Init model with new approach
model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(
    args,
    allow_auto_split = True,
    skip_load = hasattr(args, 'stream_layers') and args.stream_layers,
    benchmark = True,
    max_output_len = hasattr(args, 'max_output_len') and args.max_output_len,
    progress = True
)
cache = None

# Auto split
if not model.loaded and not (hasattr(args, 'stream_layers') and args.stream_layers):

    if hasattr(args, 'mix_layers') and args.mix_layers:
        print(" !! Warning, auto split does not account for VRAM requirement of replacement layers")

    print(" -- Loading model...")
    cache = ExLlamaV2Cache_Q4(model, lazy = True)
    t = time.time()
    model.load_autosplit(cache, progress = True)
    t = time.time() - t
    print(f" -- Loaded model in {t:.4f} seconds")

if hasattr(args, 'stream_layers') and args.stream_layers:

    stream_batch_size = 2
    model.config.max_batch_size = stream_batch_size
    model.load(lazy = True)

# Rank reduction
if hasattr(args, 'rank_reduce') and args.rank_reduce:

    if hasattr(args, 'stream_layers') and args.stream_layers:
        print(" ## --rank_reduce can not be combined with --stream_layers")
        sys.exit()

    rr = args.rank_reduce.split(",")
    idx = len(model.modules) - 1
    for r in rr:
        k = float(r)

        while True:
            idx -= 1
            module = model.modules[idx]
            if isinstance(module, ExLlamaV2ParallelDecoder): break
            if isinstance(module, ExLlamaV2MLP): break
            if isinstance(module, ExLlamaV2MoEMLP): break
            if idx < 0:
                print(" ## Not enough layers")
                sys.exit()

        print(f" -- Reducing {module.key} ({module.name}) to {k * 100:.2f}%")
        module.rank_reduce(k)

# Replacement
if hasattr(args, 'mix_layers') and args.mix_layers:
    intervals_, extra_dir = args.mix_layers.split(":")

    print(f" -- Loading replacement layers from: {extra_dir}")

    extra_config = ExLlamaV2Config()
    extra_config.model_dir = extra_dir
    extra_config.prepare()
    intervals = intervals_.split(",")
    for interval in intervals:
        ab = interval.split("-")
        a, b = int(ab[0]), int(ab[-1])
        for idx in range(a, b + 1):
            print(f" --   Layer {idx}...")
            layerkey = "model.layers." + str(idx) + "."
            remove = [k for k in model.config.tensor_file_map.keys() if k.startswith(layerkey)]
            replace = [极 for k in extra_config.tensor_file_map.keys() if k.startswith(layerkey)]
            for k in remove: del model.config.tensor_file_map[k]
            for k in replace: model.config.tensor_file_map[k] = extra_config.tensor_file_map[k]
            if not (hasattr(args, 'stream_layers') and args.stream_layers):
                model.modules[idx * 2 + 1].reload()
                model.modules[idx * 2 + 2].reload()

# Set up draft model if using speculative decoding
if use_draft_model:
    print("Setting up draft model for speculative decoding...")
    draft_config = ExLlamaV2Config(draft_model_dir)
    draft_config.scale_alpha_value = 6.0
    draft_config.max_seq_len = max_context
    draft_model = ExLlamaV2(draft_config)

    draft_cache = ExLlamaV2Cache_Q4(
        draft_model,
        max_seq_len = total_context,
        lazy = True
    )

    draft_model.load_autosplit(draft_cache, progress = True)
else:
    draft_model = None
    draft_cache = None

print("Model initialization complete")
hf_tokenizer_kwargs = {}
hf_tokenizer_kwargs.setdefault("padding_side", "left")
hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, **hf_tokenizer_kwargs)
def get_messages(messages):
    output = []
    for message in messages:
        output.append({"role": message.role, "content": message.content})
    return output

# Model Merge
#model = ExLlamaV2MergePassthrough(model)

#lora_directory = "../Documents/trained_llama3_lr2e4_r64/"
#lora = ExLlamaV2Lora.from_directory(model, lora_directory)
lora = None

#cache = ExLlamaV2Cache_Q4(
#    model,
#    max_seq_len = total_context,
    #lazy = True
#)

# Initialize the generator
print("Getting generator")

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    draft_model = draft_model,
    draft_cache = draft_cache,
    tokenizer = tokenizer,
    max_batch_size = max_batch_size,
    use_ngram_draft = use_ngram,
    max_chunk_size = max_chunk_size,
    paged = paged,
)
print("Got generator")
if lora is not None:
    generator.set_loras(lora)

# Active sequences and corresponding caches and settings
prompts = queue.Queue()
responses = {}
prompt_length = {}
# Global variable for storing partial responses
partial_responses = {}

# Create jobs
STATUS_LINES = term.height-8  # Number of lines to dedicate for status messages
LLM_LINES = max_batch_size
status_area = StatusArea(STATUS_LINES)
displays = {}
prompt_ids2jobs = {}
cancelled_request_ids = []