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
parser.add_argument('--max_chunk_size', type=int, default=2048, help='Max chunk size.')
parser.add_argument('--max_new_tokens', type=int, default=2048, help='Max new tokens.')
parser.add_argument('--use_draft_model', action="store_true", help='Do speculative decoding')
parser.add_argument('--not_paged', action="store_true", help='Do not do paged attention')

# Add arguments from the new model loading code
parser.add_argument("-ed", "--eval_dataset", type=str, help="Perplexity evaluation dataset (.parquet file)")
parser.add_argument("-er", "--eval_rows", type=int, default=None, help="Number of rows to apply from dataset (default 128)")
parser.add_argument("-el", "--eval_length", type=int, default=2048, help="Max no. tokens per sample")
parser.add_argument("-et", "--eval_token", action="store_true", help="Evaluate perplexity on token-by-token inference using cache")
parser.add_argument("-e8", "--eval_token_8bit", action="store_true", help="Evaluate perplexity on token-by-token inference using 8-bit (FP8) cache")
parser.add_argument("-eq4", "--eval_token_q4", action="store_true", help="Evaluate perplexity on token-by-token inference using Q4 cache")
parser.add_argument("-eq6", "--eval_token_q6", action="store_true", help="Evaluate perplexity on token-by-token inference using Q6 cache")
parser.add_argument("-eq8", "--eval_token_q8", action="store_true", help="Evaluate perplexity on token-by-token inference using Q8 cache")
parser.add_argument("-ecl", "--eval_context_lens", action="store_true", help="Evaluate perplexity at range of context lengths")
parser.add_argument("-p", "--prompt", type=str, help="Generate from prompt (basic sampling settings)")
parser.add_argument("-pnb", "--prompt_no_bos", action="store_true", help="Don't add BOS token to prompt")
parser.add_argument("-t", "--tokens", type=int, default=128, help="Max no. tokens")
parser.add_argument("-ps", "--prompt_speed", action="store_true", help="Test prompt processing (batch) speed over context length")
parser.add_argument("-s", "--speed", action="store_true", help="Test raw generation speed over context length")
parser.add_argument("-mix", "--mix_layers", type=str, help="Load replacement layers from secondary model. Example: --mix_layers 1,6-7:/mnt/models/other_model")
parser.add_argument("-nwu", "--no_warmup", action="store_true", help="Skip warmup before testing model")
parser.add_argument("-sl", "--stream_layers", action="store_true", help="Load model layer by layer (perplexity evaluation only)")
parser.add_argument("-sp", "--standard_perplexity", choices=["wiki2"], help="Run standard (HF) perplexity test, stride 512 (experimental)")
parser.add_argument("-rr", "--rank_reduce", type=str, help="Rank-reduction for MLP layers of model, in reverse order (for experimentation)")
parser.add_argument("-mol", "--max_output_len", type=int, help="Set max output chunk size (incompatible with ppl tests)")

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

        self.collected_output += r.get("极text", "").replace("\n", "\\n")

        token_ids = r.get("token_ids", None)
        if token_ids is not None: self.tokens += token_ids.shape[-1]

        self.prefill = r.get("curr_progress", self.prefill)
        self.max_prefill = r.get("极progress", self.max_prefill)

        text = term.black(f"{self.console_line:3}:")
        text += term.blue(f"{stage:16}")
        text += "prefill [ " + term.yellow(f"{self.prefill: 5} / {self.max_prefill: 5}")+" ]"
        text += "   "
        text += term.green(f"{self.tokens: 5} t")
       极 += term.black(" -> ")
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
    benchmark = False,
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

print("*** Loaded.. now Inference...:")
class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        print(f"Incoming request: {request.method} {request.url}")
        response = await call_next(request)
        print(f"Outgoing response status: {response.status_code}")
        return response

# take from https://github.com/tiangolo/fastapi/discussions/11360
class RequestCancelledMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global prompt_ids2jobs, prompt_length, cancelled_request_ids
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Let's make a shared queue for the request messages
        req_queue = asyncio.Queue()
        #cancelled_request_ids = []
        async def message_poller(sentinel, handler_task):
            nonlocal req_queue
            request_id = str(generate_unique_id())
            while True:
                message = await receive()
                #print(message)
                if "body" in message:
                    scope['extensions'] = {'request_id': request_id}

                if message["type"] == "http.disconnect":
                    cancelled_request_ids.append(request_id)
                    handler_task.cancel()
                    return sentinel # Break the loop

                # Puts the message in the queue
                await req_queue.put(message)

        sentinel = object()
        handler_task = asyncio.create_task(self.app(scope, req_queue.get, send))
        asyncio.create_task(message_poller(sentinel, handler_task))

        try:
            return await handler_task
        except asyncio.CancelledError:
            status_area.update(f"Cancelling request due to disconnect prompt", line=STATUS_LINES-1)
#             # TODO: FIgure out how to get prompt id that disconnected
#             while len(cancelled_request_ids) > 0:
#                 cancelled_id = cancelled_request_ids.pop()
#                 if cancelled_id in prompt_ids2jobs:
#                     generator.cancel(prompt_ids2jobs[cancelled_id])
#                     del prompt_ids2jobs[cancelled_id]
#                     del prompt_length[cancelled_id]
#                     status_area.update(f"Cancelling request due to disconnect prompt: {cancelled_id}", line=STATUS_LINES-1)
#                 else: 
#                     status_area.update(f"Cannot find job: {cancelled_id}", line=STATUS_LINES-1)


app = FastAPI(title="EXL2")
app.add_middleware(RequestLoggerMiddleware)
app.add_middleware(RequestCancelledMiddleware)

async def stream_response(prompt_id, timeout=180):
    global partial_responses
    while True:
        await asyncio.sleep(0.05)  # Sleep to yield control to the event loop

        # Check if prompt_id exists in partial_responses
        if prompt_id in partial_responses:
            # Stream partial responses
            while partial_responses[prompt_id]:
                response_chunk = partial_responses[prompt_id].pop(0)
                yield f"data: {json.dumps(response_chunk)}\n\n"

            # Check for final response or timeout
            if prompt_id in responses:
                final_response = responses.pop(prompt_id)
                yield f'data: {{"id":"chatcmpl-{prompt_id}","object":"chat.completion.chunk","created":{int(time.time())},"model":"{repo_str}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}\n\n'
                break


def process_prompts():
    global partial_responses
    global prompt_ids2jobs, prompt_length, cancelled_request_ids
    try:

        while True:
            while not prompts.empty() or len(prompt_length):
                while len(prompt_length) < max_batch_size and not prompts.empty():
                    prompt_id, prompt, max_tokens, stream, temperature, outlines_dict = prompts.get()
                    stop = outlines_dict.get("stop", None)
                    if outlines_dict["type"] == "choices":
                        filters = [ChoiceFilter(outlines_dict["choices"], hf_tokenizer)]
                    elif outlines_dict["type"] == "json":
                        filters = [JSONFilter(outlines_dict["json"], hf_tokenizer)]
                    elif outlines_dict["type"] == "regex":
                        # Validation of regex
                        filters = [RegexFilter(outlines_dict["regex"], hf_tokenizer)]
                    else:
                        filters = []
                    ids = tokenizer.encode(prompt, encode_special_tokens = True)
                    prompt_tokens = ids.shape[-1]
                    new_tokens = prompt_tokens + max_tokens
                    #print("Processing prompt: " + str(prompt_id) + "  Req tokens: " + str(new_tokens))
                    status_area.update(f"Processing prompt: {prompt_id}  Req tokens: {new_tokens}", line=STATUS_LINES-1)
                    # Truncate if new_tokens exceed max_context
                    if new_tokens > max_context:
                        # Calculate how many tokens to truncate
                        ids = tokenizer.encode("Say, 'Prompt exceeds allowed length. Please try again.'")
                        # Update new_tokens after truncation
                        prompt_tokens = ids.shape[-1]
                        new_tokens = prompt_tokens + max_tokens
                        print("Truncating prompt: " + str(prompt_id) + "  Req tokens: " + str(new_tokens))
                    prompt_length[prompt_id] = prompt_tokens
                    #streamer.append(stream)
                    #prompt_ids.append(prompt_id)

                    eos_token_ids = [tokenizer.eos_token_id, hf_tokenizer.eos_token_id]
                    # print("eos token ids", eos_token_ids)
                    if config_eos_token_ids is not None:
                        eos_token_ids.extend([int(c) for c in config_eos_token_ids.split(',')])
                    if stop is not None:
                        if isinstance(stop, list):
                            for stop_string in stop:
                                eos_token_ids.append(stop_string)
                        else:
                            eos_token_ids.append(stop)  
                    gen_settings = ExLlamaV2Sampler.Settings()
                    gen_settings.temperature = 2.0 if temperature>2 else temperature  # To make sure the temperature value does not exceed 2

                    job = ExLlamaV2DynamicJob(
                        input_ids = ids,
                        max_new_tokens = max_tokens,
                        stop_conditions = eos_token_ids,
                        gen_settings = gen_settings,
                        filters = filters,
                        token_healing = healing
                    )

                    job.prompt_length = prompt_tokens
                    job.input_ids = ids
                    job.streamer = stream
                    job.prompt_ids = prompt_id
                    job.stop = stop

                    generator.enqueue(job)
                    #displays = { job: JobStatusDisplay(job, line, STATUS_LINES) for line, job in enumerate(jobs) }
                    displays[job] = JobStatusDisplay(job, STATUS_LINES)

                    for index, (job, display) in enumerate(list(displays.items())):
                        display.update_position(index%LLM_LINES)  # Set position before updating
                    prompt_ids2jobs[prompt_id] = job

                if(len(prompt_length)):
                    results = generator.iterate()
                    for r in results:
                        job = r["job"]
                        displays[job].update(r)
                        displays[job].display()
                        stage = r["stage"]
                        stage = r.get("eos_reason", stage)
                        outcontent = r.get("text", "")
                        reason = None
                        if(job.streamer):

                            partial_response_data = {
                                "id": f"chatcmpl-{job.prompt_ids}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": repo_str,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": outcontent
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }

                            # Initialize a list for new prompt_id or append to existing one
                            if job.prompt_ids not in partial_responses:
                                partial_responses[job.prompt_ids] = []
                            partial_responses[job.prompt_ids].append(partial_response_data)

                        if r['eos'] == True:
                            total_time = r['time_generate']
                            total_tokens = r['new_tokens']
                            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
                            status_area.update(f"EOS detected: {stage}, Generated Tokens: {total_tokens}, Tokens per second: {tokens_per_second}/s", line=STATUS_LINES-2)

                            #generated_part = job.input_ids[:, job.prompt_length:]
                            #output = tokenizer.decode(generated_part[0]).strip()
                            #output = tokenizer.decode(input_ids[i])[0]
                            generated_text = r['full_completion']

                            # Calculate token counts
                            completion_tokens_old = (tokenizer.encode(generated_text)).shape[-1]

                            completion_tokens = r['new_tokens']
                            prompt_tokens = r['prompt_tokens']

                            full_tokens = completion_tokens + prompt_tokens
                            status_area.update(f"Completion Tokens: {completion_tokens_old}, New Completion Tokens: {completion_tokens}", line=STATUS_LINES-3)


                            eos_prompt_id = job.prompt_ids
                            if(job.streamer):
                                ## Generator, yield here..
                                partial_response_data = {
                                    "finish_reason": "stop"
                                }

                                responses[eos_prompt_id] = partial_response_data
                            else:# Construct the response based on the format
                                response_data = {
                                    "id": f"chatcmpl-{eos_prompt_id}",
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": repo_str,
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": generated_text,
                                        },
                                        "finish_reason": "stop"
                                    }],
                                    "usage": {
                                        "prompt_tokens": prompt_tokens,
                                        "completion_tokens": completion_tokens,
                                        "total_tokens": full_tokens
                                    }
                                }
                                responses[eos_prompt_id] = response_data
                            del prompt_ids2jobs[eos_prompt_id]
                            del prompt_length[eos_prompt_id]
                    if len(cancelled_request_ids):
                        cancelled_id = cancelled_request_ids.pop()
                        status_area.update(f"Cancelling request due to disconnect prompt: {cancelled_id}", line=STATUS_LINES-1)
                        if cancelled_id in prompt_ids2jobs:
                            generator.cancel(prompt_ids2jobs[cancelled_id])
                            del prompt_ids2jobs[cancelled_id]
                            del prompt_length[cancelled_id]
                            status_area.update(f"Found and cancelling: {cancelled_id}", line=STATUS_LINES-1)
                        else: 
                            # Temporarily store items to check against cancelled_id
                            temp_storage = []

                            # Drain the queue and check each item
                            while not prompts.empty():
                                prompt_id, prompt, max_tokens, stream, temperature, outlines_dict = prompts.get()
                                if prompt_id != cancelled_id:
                                    # Only requeue prompts that do not match the cancelled_id
                                    temp_storage.append((prompt_id, prompt, max_tokens, stream, temperature, outlines_dict))

                            # Re-add the valid items back to the queue
                            for item in temp_storage:
                                prompts.put(item)



            else:
                # Sleep for a short duration when there's no work
                time.sleep(0.1)  # Sleep for 100 milliseconds
    except Exception as e:
        print("Reset server due to ", e)
        print(traceback.format_exc())
        for prompt_id in prompt_ids2jobs:
            job = prompt_ids2jobs[prompt_id]
            if(job.streamer):
                ## Generator, yield here..
                partial_response_data = {
                    "finish_reason": "stop"
                }

                responses[prompt_id] = partial_response_data
            else:
                print("Error handling for full generation current not implemented")
            generator.cancel(job)
        prompt_ids2jobs = {}
        prompt_length = {}

# Start worker thread
worker = Thread(target=process_prompts)
worker.start()

async def do_completion(requestid: Request, prompt: str, request):
    try:
        timeout = 180  # seconds
        start_time = time.time()
        prompt_id = requestid.scope.get("extensions", {}).get("request_id", "Unknown ID")
        #prompt_id = generate_unique_id()
        status_area.update(f"Prompt: {prompt}, Prompt ID: {prompt_id}")
        outlines_dict = {}
        
        # Adjust temperature if it is 0
        if request.temperature == 0:
            request.temperature = 0.001

        if request.stop is not None:
            outlines_dict["stop"] = request.stop
        if request.outlines_type is not None:
            outlines_dict["type"] = request.outlines_type
        else:
            outlines_dict["type"] = "text"
        if outlines_dict["type"] == "choices":
            assert request.choices is not None
            outlines_dict["choices"] = request.choices
        elif outlines_dict["type"] == "json":
            assert request.json is not None
            outlines_dict["json"] = request.json
        elif outlines_dict["type"] == "regex":
            assert request.regex is not None
            outlines_dict["regex"] = request.regex
        else:
            assert outlines_dict["type"] == "text"
        prompts.put((prompt_id, prompt, request.max_tokens, request.stream, request.temperature, outlines_dict))

        if request.stream:
            #response = StreamingResponse(streaming_request(prompt, request.max_tokens, tempmodel=repo_str, response_format='chat_completion'), media_type="text/event-stream")
            return StreamingResponse(stream_response(prompt_id), media_type="text/event-stream")
        else:
            #response_data = non_streaming_request(prompt, request.max_tokens, tempmodel=repo_str, response_format='chat_completion')
            #response = response_data  # This will return a JSON response
            while prompt_id not in responses:
                await asyncio.sleep(0.1)  # Sleep to yield control to the event loop
                if (time.time() - start_time) > timeout:
                    return {"error": "Response timeout"} 

            return responses.pop(prompt_id)

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/v1/completions')
async def maincompletion(requestid: Request, request: CompletionRequest):
    print("got completions request")
    assert isinstance(request.prompt, str)
    return await do_completion(requestid, request.prompt, request)

@app.post('/v1/chat/completions')
async def mainchat(requestid: Request, request: ChatCompletionRequest):
    print("got chat completions request")
    hf_get_messages = get_messages(request.messages)
    prompt = hf_tokenizer.apply_chat_template(hf_get_messages, tokenize=False, add_generation_prompt=True)
    if request.partial_generation is not None:
        prompt += request.partial_generation
    return await do_completion(requestid, prompt, request)





@app.get('/ping')
async def get_status():
    return {"ping": sum(prompt_length.values())}

@app.get("/nvidia-smi")
async def get_nvidia_smi():
    # Execute the nvidia-smi command
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    nvidia_smi_output = result.stdout.strip()  # Remove any extra whitespace
    # Split the output by lines and then by commas
    gpu_data = []
    for line in nvidia_smi_output.split("\n"):
        utilization, memory_used, memory_total = line.split(", ")
        # Strip the '%' and 'MiB' and convert to appropriate types
        utilization = float(utilization.strip(' %'))
        memory_used = int(memory_used.strip(' MiB'))
        memory_total = int(memory_total.strip(' MiB'))
        gpu_data.append({
           "utilization": utilization,
           "memory_used": memory_used,
           "memory_total": memory_total
        })
    return gpu_data


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port, log_level="error")
    print(term.enter_fullscreen())