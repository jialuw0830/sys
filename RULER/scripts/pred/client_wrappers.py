# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import abc
import json
import multiprocessing
import os
import re
import sys
import time
import requests
import traceback
from pathlib import Path
from typing import List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 


class Client(abc.ABC):
    def __init__(
        self,
        server_host,
        server_port='5000',
        ssh_server=None,
        ssh_key_path=None,
        **generation_kwargs
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.ssh_server = os.getenv("SSH_SERVER", ssh_server)
        self.ssh_key_path = os.getenv("SSH_KEY_PATH", ssh_key_path)
        self.generation_kwargs = generation_kwargs
        
    @abc.abstractmethod
    def _single_call(
        self,
        prompts,
    ):
        pass

    def __call__(
        self,
        prompt: str,
        **kwargs
    ):
        request = self.generation_kwargs
        # prompts are added later
        request['prompts'] = [f'{prompt}']
        if 'others' in kwargs:
            request['others'] = kwargs['others']

        outputs = self._single_call(**request)
        response = {'text': outputs}
        return response
        
    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request, route="generate"):
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(f"ssh://{self.ssh_server}:22", self.ssh_key_path)
            outputs = sshtunnel_request.put(
                url="http://{}:{}/{}".format(self.server_host, self.server_port, route),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
        else:
            outputs = requests.put(
                url="http://{}:{}/{}".format(self.server_host, self.server_port, route),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
            ).json()
        return outputs

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        num_threads = max(96, multiprocessing.cpu_count() * 16)
        with ThreadPoolExecutor(num_threads) as executor:
            futures = []
            for prompt in prompts:
                futures.append(
                    executor.submit(
                        self.__call__,
                        prompt,
                        **kwargs,
                    )
                )
            rets = [f.result() for f in futures]
        return rets


class TRTLLMClient(Client):
    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        random_seed,
        stop: List[str],
        max_attention_window_size=None,
    ):
        request = {
            "prompts": prompts,
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "random_seed": random_seed,
            'stop_words_list': ",".join(stop),
        }
        if max_attention_window_size:
            request["max_attention_window_size"] = max_attention_window_size
            
        outputs = self._send_request(request)
        return outputs


class VLLMClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._tokenizer_loaded = False

    def _maybe_load_tokenizer(self):
        if self._tokenizer_loaded:
            return
        self._tokenizer_loaded = True

        tokenizer_name = self.generation_kwargs.get("tokenizer_name_or_path")
        if not tokenizer_name:
            return

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback to whitespace-based approximation if tokenizer loading fails.
            self._tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0

        self._maybe_load_tokenizer()
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass

        return len(text.split())

    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request_stream(self, request, route="generate"):
        if self.ssh_server and self.ssh_key_path:
            import sshtunnel_requests

            sshtunnel_request = sshtunnel_requests.from_url(
                f"ssh://{self.ssh_server}:22", self.ssh_key_path
            )
            response = sshtunnel_request.put(
                url="http://{}:{}/{}".format(self.server_host, self.server_port, route),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
                stream=True,
            )
        else:
            response = requests.put(
                url="http://{}:{}/{}".format(self.server_host, self.server_port, route),
                data=json.dumps(request),
                headers={"Content-Type": "application/json"},
                stream=True,
            )

        response.raise_for_status()
        return response

    def _single_call_stream(
        self,
        request,
    ):
        start_t = time.perf_counter()
        response = self._send_request_stream(request)
        prompt_text = request.get("prompt", "")

        first_token_t = None
        chunk_times = []
        generated_tokens = 0
        current_text = ""

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue

            try:
                line = raw_line.decode("utf-8").strip()
            except Exception:
                continue

            if not line:
                continue

            if line.startswith("data:"):
                line = line[5:].strip()

            if line == "[DONE]":
                break

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            text_field = payload.get("text", "")
            if isinstance(text_field, list):
                new_text = text_field[0] if len(text_field) > 0 else ""
            else:
                new_text = text_field

            if new_text is None:
                continue

            if isinstance(prompt_text, str) and prompt_text and new_text.startswith(prompt_text):
                new_text = new_text[len(prompt_text):]

            if new_text.startswith(current_text):
                delta_text = new_text[len(current_text):]
            else:
                delta_text = new_text

            if delta_text:
                now_t = time.perf_counter()
                if first_token_t is None:
                    first_token_t = now_t
                chunk_times.append(now_t)
                generated_tokens += self._count_tokens(delta_text)

            current_text = new_text

        end_t = time.perf_counter()

        ttft = None
        if first_token_t is not None:
            ttft = first_token_t - start_t

        itls = []
        if len(chunk_times) >= 2:
            for idx in range(1, len(chunk_times)):
                itls.append(chunk_times[idx] - chunk_times[idx - 1])

        duration = max(end_t - start_t, 1e-9)
        request_throughput = generated_tokens / duration
        request_samples_per_second = 1.0 / duration

        stream_metrics = {
            "ttft": ttft,
            "itl": (sum(itls) / len(itls)) if len(itls) > 0 else None,
            "itl_sum": sum(itls),
            "itl_count": len(itls),
            "generated_tokens": generated_tokens,
            "request_duration": duration,
            "request_throughput": request_throughput,
            "request_samples_per_second": request_samples_per_second,
            "request_start": start_t,
            "request_end": end_t,
        }

        return {"text": [current_text], "stream_metrics": stream_metrics}

    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        random_seed,
        stop: List[str],
    ):
        request = {
            "prompt": prompts[0],
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stop": stop,
        }
        # TODO: random seed is not supported?
        stream = bool(self.generation_kwargs.get("stream", False))
        if stream:
            return self._single_call_stream(request)

        outputs = self._send_request(request)
        outputs = outputs['text']
        return outputs

    def __call__(
        self,
        prompt: str,
        **kwargs
    ):
        request = self.generation_kwargs
        request['prompts'] = [f'{prompt}']
        if 'others' in kwargs:
            request['others'] = kwargs['others']

        outputs = self._single_call(**request)
        if isinstance(outputs, dict) and 'text' in outputs:
            return outputs

        response = {'text': outputs}
        return response


class SGLClient(Client):
    def _single_call(
        self,
        prompts,
        tokens_to_generate,
        temperature,
        top_p,
        top_k,
        random_seed,
        stop: List[str],
    ):
        request = {
            "text": prompts[0],
            "sampling_params": {
                "max_new_tokens": tokens_to_generate,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "stop": stop,
            }
        }
        # TODO: random seed is not supported?
        outputs = self._send_request(request)
        outputs = outputs['text']
        return outputs


class OpenAIClient:
    def __init__(
        self,
        model_name,
        base_url=None,
        api_key=None,
        **generation_kwargs
    ):  
        model2length = {
            # OpenAI
            'gpt-4': 8192,
            'gpt-4-0613': 8192,
            'gpt-4-1106-preview': 128000,
            'gpt-4-0125-preview': 128000,
            'gpt-4-turbo-preview': 128000,
            'gpt-3.5-turbo-0125': 16385,
            'gpt-3.5-turbo-1106': 16385,
            'gpt-3.5-turbo-0613': 4096,
            'gpt-3.5-turbo': 16385,
            'gpt-3.5-turbo-16k': 16385,
            'gpt-3.5-turbo-16k-0613': 16385,

            # Azure
            'gpt-4-32k': 32768,
            'gpt-4': 128000,
            'gpt-35-turbo-16k': 16384,
        }
        self.base_url = base_url
        self.openai_api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.azure_api_id = os.getenv("AZURE_API_ID", "")
        self.azure_api_secret = os.getenv("AZURE_API_SECRET", "")
        self.azure_api_endpoint = os.getenv("AZURE_API_ENDPOINT", "")
        self.model_name = model_name    
        # Default behavior: disable model thinking/reasoning mode unless explicitly enabled.
        self.enable_thinking = bool(generation_kwargs.pop("enable_thinking", False))
        # For OpenAI-compatible serving (e.g. vLLM), prefer official chat.completions path.
        self.use_chat_completions = bool(generation_kwargs.pop("use_chat_completions", True))
            
        # Azure
        if self.azure_api_id and self.azure_api_secret:
            if 'gpt-3.5' in model_name: self.model_name = 'gpt-35-turbo-16k'
            if 'gpt-4' in model_name: self.model_name = 'gpt-4'
        
        import tiktoken
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_length = model2length.get(self.model_name, 131072)
        self.generation_kwargs = generation_kwargs
        self._create_client()
        
    def _create_client(self,):
        from openai import OpenAI, AzureOpenAI

        if self.base_url:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.openai_api_key or "token-abc123",
            )
            return
        
        # OpenAI
        if self.openai_api_key:
            self.client = OpenAI(
                api_key=self.openai_api_key
            )

        # Azure
        elif self.azure_api_id and self.azure_api_secret:
            self.client = AzureOpenAI(
                api_key=self.get_azure_api_key(
                    self.azure_api_id, 
                    self.azure_api_secret,
                    self.azure_api_endpoint,
                ),
                api_version="2024-02-15-preview",
                azure_endpoint=os.path.join(self.azure_api_endpoint, "llm/v1/azure"),
            )
        
    def _count_tokens(self, messages):
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
        
    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request):
        if self.base_url:
            extra_body = {
                "top_k": request['top_k'],
                "seed": request['random_seed'],
                "enable_thinking": self.enable_thinking,
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                },
            }
            if self.use_chat_completions:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": request['prompt']}],
                    max_tokens=request['tokens_to_generate'],
                    temperature=request['temperature'],
                    top_p=request['top_p'],
                    stop=request['stop'],
                    extra_body=extra_body,
                )
            else:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=request['prompt'],
                    max_tokens=request['tokens_to_generate'],
                    temperature=request['temperature'],
                    top_p=request['top_p'],
                    stop=request['stop'],
                    extra_body=extra_body,
                )
            return response

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=request['msgs'],
                max_tokens=request['tokens_to_generate'],
                temperature=request['temperature'],
                seed=request['random_seed'],
                top_p=request['top_p'],
                stop=request['stop'],
            )
        except Exception as e:
            print(f"Error occurred while calling OpenAI: {e}")
            if self.azure_api_id and self.azure_api_secret and e.status_code == 401:
                # token expired
                self._create_client()
            
        return response

    @retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(3))
    def _send_request_stream(self, request):
        extra_body = {
            "top_k": request['top_k'],
            "seed": request['random_seed'],
            "enable_thinking": self.enable_thinking,
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
            },
        }
        if self.use_chat_completions:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": request['prompt']}],
                max_tokens=request['tokens_to_generate'],
                temperature=request['temperature'],
                top_p=request['top_p'],
                stop=request['stop'],
                stream=True,
                extra_body=extra_body,
            )
        else:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=request['prompt'],
                max_tokens=request['tokens_to_generate'],
                temperature=request['temperature'],
                top_p=request['top_p'],
                stop=request['stop'],
                stream=True,
                extra_body=extra_body,
            )
        return response

    def _count_tokens_text(self, text):
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text.split())

    def _single_call_stream(self, request):
        start_t = time.perf_counter()
        first_token_t = None
        chunk_times = []
        generated_tokens = 0
        current_text = ""

        stream = self._send_request_stream(request)
        for chunk in stream:
            if not hasattr(chunk, 'choices') or len(chunk.choices) == 0:
                continue

            choice = chunk.choices[0]
            delta_text = ""
            if hasattr(choice, 'delta') and getattr(choice, 'delta') is not None:
                delta_content = getattr(choice.delta, 'content', None)
                if isinstance(delta_content, str):
                    delta_text = delta_content
                elif isinstance(delta_content, list):
                    parts = []
                    for item in delta_content:
                        if isinstance(item, dict):
                            text_part = item.get('text')
                            if isinstance(text_part, str):
                                parts.append(text_part)
                    delta_text = "".join(parts)
            if not delta_text and hasattr(choice, 'text'):
                delta_text = choice.text or ""
            if not delta_text:
                continue

            now_t = time.perf_counter()
            if first_token_t is None:
                first_token_t = now_t
            chunk_times.append(now_t)
            generated_tokens += self._count_tokens_text(delta_text)
            current_text += delta_text

        end_t = time.perf_counter()

        ttft = None
        if first_token_t is not None:
            ttft = first_token_t - start_t

        itls = []
        if len(chunk_times) >= 2:
            for idx in range(1, len(chunk_times)):
                itls.append(chunk_times[idx] - chunk_times[idx - 1])

        duration = max(end_t - start_t, 1e-9)
        stream_metrics = {
            "ttft": ttft,
            "itl": (sum(itls) / len(itls)) if len(itls) > 0 else None,
            "itl_sum": sum(itls),
            "itl_count": len(itls),
            "generated_tokens": generated_tokens,
            "request_duration": duration,
            "request_throughput": generated_tokens / duration,
            "request_samples_per_second": 1.0 / duration,
            "request_start": start_t,
            "request_end": end_t,
        }
        return {"text": [current_text], "stream_metrics": stream_metrics}
        
    def __call__(
        self,
        prompt: str,
    ):
        if self.base_url:
            request = self.generation_kwargs
            request["prompt"] = prompt
            if bool(request.get('stream', False)):
                return self._single_call_stream(request)
            outputs = self._send_request(request)
            if self.use_chat_completions:
                return {'text': [outputs.choices[0].message.content]}
            return {'text': [outputs.choices[0].text]}

        # system_msg = [{"role": "system", "content": ""}]
        system_msg = []
        user_assistant_msgs = [{"role": "user", "content": prompt}]
        msgs = system_msg + user_assistant_msgs
        openai_length = self._count_tokens(msgs)
        request = self.generation_kwargs
        
        tokens_to_generate_new = self.max_length - openai_length
        if tokens_to_generate_new < request['tokens_to_generate']:
            print(f"Reduce generate tokens from {request['tokens_to_generate']} to {tokens_to_generate_new}")
            request['tokens_to_generate'] = tokens_to_generate_new
    
        request["msgs"] = msgs
        outputs = self._send_request(request)
        response = {'text': [outputs.choices[0].message.content]}
        return response

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        num_threads = max(96, multiprocessing.cpu_count() * 16)
        with ThreadPoolExecutor(num_threads) as executor:
            futures = []
            for prompt in prompts:
                futures.append(executor.submit(self.__call__, prompt))
            rets = [f.result() for f in futures]
        return rets

    
    def get_azure_api_key(
        self,
        p_client_id, 
        p_client_secret, 
        p_token_url, 
        p_scope="azureopenai-readwrite",
        cache_file="azure_openai_key.json"
    ):
        base_path = Path(__file__).parent
        file_path = Path.joinpath(base_path, cache_file)
     
        # Check if the token is cached
        renew = True
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                token = json.load(f)
                renew = True if time.time() > token["expires_in"] else False

        if renew:
            # Get a new token from the OAuth server
            response = requests.post(
                os.path.join(p_token_url, "oauth/api/v1/ssa/default/token"),
                data={"grant_type": "client_credentials", "client_id": p_client_id,
                        "client_secret": p_client_secret, "scope": p_scope}
            )
            response.raise_for_status()
            token = response.json()
            token["expires_in"] += time.time()
            with open(file_path, "w") as f:
                json.dump(token, f)
     
     
        authToken = token["access_token"]
        return authToken


class GeminiClient:
    def __init__(
        self,
        model_name,
        **generation_kwargs
    ):
        model2length = {
            'gemini-1.0-pro-latest': (30720, 2048),
            'gemini-1.5-pro-latest': (1048576, 8192)
        }
        
        self.model_name = model_name
        self.model = self._initialize_model()
        self.max_input_length = model2length[model_name][0]
        self.max_output_length = model2length[model_name][1]
        assert generation_kwargs['tokens_to_generate'] < self.max_output_length, \
            print(f'tokens_to_generate exceeds {self.max_output_length}')
        
        import google.generativeai as genai        
        self.config = genai.GenerationConfig(
            candidate_count=1,
            stop_sequences=generation_kwargs['stop'],
            max_output_tokens=generation_kwargs['tokens_to_generate'],
            temperature=generation_kwargs['temperature'],
            top_p=generation_kwargs['top_p'],
            top_k=generation_kwargs['top_k'],
        )

        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    @retry(wait=wait_random_exponential(min=60, max=60), stop=stop_after_attempt(3))
    def _send_request(self, request):
        try:
            response = self.model.generate_content(request['prompt'], 
                                                   generation_config=request['config'],
                                                   safety_settings=self.safety_settings)
        except Exception as e:
            traceback.print_exc()
            return None
        return response
        
    def __call__(
        self,
        prompt: str,
    ):
        assert self.model.count_tokens(prompt).total_tokens < self.max_input_length, \
            print(f'input length exceeds {self.max_input_length}')
        
        request = {
            'prompt': prompt,
            'config': self.config,
        }
        
        outputs = self._send_request(request)

        try:
            response = {'text': [outputs.candidates[0].content.parts[0].text]}
        except Exception as e:
            response = {'text': []}
            print(outputs)
            traceback.print_exc()
            
        return response

    def _initialize_model(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        return genai.GenerativeModel(self.model_name)

