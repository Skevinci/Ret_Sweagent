# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import sys
import json
import time
import requests
from contextlib import contextmanager
from copy import deepcopy
from typing import List

import torch
import torch.distributed
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch import nn
from vllm import SamplingParams

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _pre_process_response(pad_token_id, response_token_ids: torch.Tensor) -> List[int]:
    # remove the right padding in the response token_id
    non_pad_index = torch.nonzero(response_token_ids != pad_token_id, as_tuple=False)[-1][0]
    token_ids = response_token_ids[:non_pad_index + 1].tolist()
    return token_ids

def retrieve_context_train(query, repo_commit):
    url = "https://cogcomp.seas.upenn.edu/ow4008/retrieve"
    data = {"query": query, "repo_commit": repo_commit}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    
def retrieve_context_test(query, repo_commit):
    url = "https://cogcomp.seas.upenn.edu/mo4002/retrieve"
    data = {"query": query, "repo_commit": repo_commit}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    
def batch_retrieve_train(query_list, repo_commit_list, sample_n, batch_size=128):
    contexts = []
    for i in range(0, len(query_list), batch_size):
        # start_time = time.time()
        batch_query = query_list[i:i + batch_size]
        # batch_repo_commit = repo_commit_list[i:i + batch_size]
        # batch_repo_commit is half size of query
        batch_repo_commit = [repo_commit_list[j // sample_n] for j in range(i, i + len(batch_query))]
        
        # Call the retrieve_context_rm function for each batch
        batch_contexts = json.loads(retrieve_context_train(batch_query, batch_repo_commit))
        if batch_contexts:
            contexts.extend(batch_contexts)
        # end_time = time.time()
        # print(f"Time taken for batch {i} retrieval: {end_time - start_time} seconds")
    
    return contexts

def batch_retrieve_test(query_list, repo_commit_list, batch_size=128):
    contexts = []
    for i in range(0, len(query_list), batch_size):
        # start_time = time.time()
        batch_query = query_list[i:i + batch_size]
        batch_repo_commit = repo_commit_list[i:i + batch_size]
        
        # Call the retrieve_context_rm function for each batch
        batch_contexts = retrieve_context_test(batch_query, batch_repo_commit)
        if batch_contexts:
            contexts.extend(batch_contexts)
        # end_time = time.time()
        # print(f"Time taken for batch {i} retrieval: {end_time - start_time} seconds")
    
    return contexts


class vLLMRollout(BaseRollout):
    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = int(self.config.get("max_num_batched_tokens", 8192))

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp")
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config else OmegaConf.to_container(deepcopy(config.engine_kwargs))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in (
            "0.5.4",
            "0.6.3",
        ):
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        
        repo_list = prompts.non_tensor_batch['repo']
        base_commit_list = prompts.non_tensor_batch['base_commit']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "max_tokens": 50,
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "max_tokens": 50,
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            kwargs = {
                "max_tokens": 50,
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False,
            )
            
            query_response = output[0].to(idx.device)
            query_text = self.tokenizer.batch_decode(query_response, skip_special_tokens=True)
            
            sample_n = self.sampling_params.n
            query_list = []
            repo_commit_list = []
            for i in range(batch_size):
                repo_commit_list.append((repo_list[i], base_commit_list[i]))
                for j in range(sample_n):
                    idx_flat = i * sample_n + j
                    query_list.append(query_txt[idx_flat])
                    
            print(f"length of query_list: {len(query_list)}")
            
            if do_sample:
                contexts = batch_retrieve_train(query_list, repo_commit_list, sample_n, batch_size=128)
            else:
                contexts = batch_retrieve_test(query_list, repo_commit_list, batch_size=128)
            context_data = self.tokenizer(contexts, return_tensors="pt", padding=True, truncation=True, max_length=5000, add_special_tokens=False).to(idx.device)
            
            patch_input_list = []
            patch_input_ids_ts_list = []
            attention_mask_list = []
            for i in range(len(contexts)):
                processed_query = _pre_process_response(self.pad_token_id, query_response[i])

                context_ids = context_data['input_ids'][i]
                context_attention_mask = context_data['attention_mask'][i]
                patch_input_ids = idx_list[i // sample_n] + processed_query + _pre_process_response(self.pad_token_id, context_ids)
                patch_input_ids_ts = torch.tensor(patch_input_ids).to(idx.device)
                attention_mask = (patch_input_ids_ts != self.pad_token_id).long()
                
                sequence_length = len(patch_input_ids_ts)
                if sequence_length < self.config.prompt_length:
                    patch_input_ids_ts = verl_F.pad_sequence_to_length(patch_input_ids_ts, self.config.prompt_length, self.pad_token_id, left_pad=True)
                    attention_mask = verl_F.pad_sequence_to_length(attention_mask, self.config.prompt_length, 0, left_pad=True)
                elif sequence_length > self.config.prompt_length:
                    patch_input_ids = patch_input_ids[:self.config.prompt_length]
                    patch_input_ids_ts = patch_input_ids_ts[:self.config.prompt_length]
                    attention_mask = attention_mask[:self.config.prompt_length]
                    
                patch_input_list.append(patch_input_ids)
                patch_input_ids_ts_list.append(patch_input_ids_ts)
                attention_mask_list.append(attention_mask)
                
            patch_idx = torch.stack(patch_input_ids_ts_list, dim=0)
            attention_mask = torch.stack(attention_mask_list, dim=0)
            position_ids = compute_position_id_with_mask(attention_mask)
            
            if not do_sample:
                kwargs = {
                    'max_tokens': self.config.response_length,
                    'best_of': 1,
                    'top_p': 1.0,
                    'top_k': -1,
                    'min_p': 0.0,
                    'temperature': 0,
                    'n': 1  # if greedy, only 1 response
                }
            else:
                kwargs = {
                    'max_tokens': self.config.response_length,
                }
                
            with self.update_sampling_params(**kwargs):
                patch_output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token ids
                    sampling_params=self.sampling_params,
                    prompt_token_ids=patch_input_list,
                    use_tqdm=False)
            
            patch_response = patch_output[0].to(idx.device)

            if patch_response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(patch_response, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.n > 1 and do_sample:
                patch_idx = patch_idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n * self.sampling_params.n
            seq = torch.cat([patch_idx, patch_response], dim=-1)

        response_length = patch_response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=patch_response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": patch_idx,
                "responses": patch_response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
