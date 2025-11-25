"""
app/llm.py
LLM client wrapper with simple support for:
 - FastChat-like chat completions endpoint (OpenAI-like /v1/chat/completions)
 - HuggingFace Text-Generation-Inference (TGI) /generate endpoint

Configure which backend to use in configs/llm_config.json.
"""
import requests
from typing import List, Dict, Any
import json
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "llm_config.json")

class LLMError(Exception):
    pass

class LLMClient:
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.backend = cfg.get("backend", "fastchat")  # "fastchat" or "hf_tgi"
        self.endpoint = cfg.get("endpoint", "http://localhost:8000")
        self.model = cfg.get("model", "")
        self.timeout = cfg.get("timeout", 30)
        # optional headers (api keys, etc.)
        self.headers = cfg.get("headers", {})

    def _call_fastchat(self, messages: List[Dict[str, str]]) -> str:
        """
        Expect a FastChat/OpenAI-compatible endpoint at {endpoint}/v1/chat/completions
        Payload: {"model": model, "messages": messages}
        """
        url = self.endpoint.rstrip("/") + "/v1/chat/completions"
        payload = {"model": self.model, "messages": messages}
        r = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
        if r.status_code != 200:
            raise LLMError(f"FastChat call failed: {r.status_code} {r.text}")
        data = r.json()
        # Try to extract the assistant text (OpenAI-like response)
        try:
            text = data["choices"][0]["message"]["content"]
            return text
        except Exception as e:
            raise LLMError(f"Unexpected FastChat response format: {e} - {data}")

    def _call_hf_tgi(self, prompt: str) -> str:
        """
        Call HuggingFace Text-Generation-Inference /generate endpoint.
        Endpoint example: http://localhost:8080
        Payload example: {"inputs": prompt, "parameters": {"max_new_tokens": 512}}
        """
        url = self.endpoint.rstrip("/") + "/generate"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 512, "temperature": 0.2}}
        r = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
        if r.status_code != 200:
            raise LLMError(f"HuggingFace TGI call failed: {r.status_code} {r.text}")
        data = r.json()
        # TGI usually returns list with 'generated_text'
        try:
            if isinstance(data, list):
                return data[0].get("generated_text", "")
            return data.get("generated_text", "")
        except Exception as e:
            raise LLMError(f"Unexpected HF-TGI response format: {e} - {data}")

    def generate_chat(self, system_prompt: str, history: List[Dict[str, str]]) -> str:
        """
        Build an LLM request from system prompt and history (list of {"role":..., "content":...})
        and return assistant text.
        """
        if self.backend == "fastchat":
            # Build messages for OpenAI-like API
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            return self._call_fastchat(messages)
        elif self.backend == "hf_tgi":
            # Build a single prompt by concatenating system + history with roles
            parts = []
            if system_prompt:
                parts.append("System: " + system_prompt.strip())
            for msg in history:
                role = msg["role"].capitalize()
                parts.append(f"{role}: {msg['content'].strip()}")
            parts.append("Assistant:")
            prompt = "\n".join(parts)
            return self._call_hf_tgi(prompt)
        else:
            raise LLMError(f"Unsupported backend: {self.backend}")