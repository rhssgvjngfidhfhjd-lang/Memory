import threading
import time
from abc import ABC, abstractmethod
from typing import List, Sequence

class BaseLLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleBackend(BaseLLMBackend):
    def __init__(
        self,
        model: str,
        api_base: str,
        api_key,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_p: float = 1.0,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
        timeout: int = 60,
    ):
        keys = _normalize_api_keys(api_key)
        if not keys:
            raise ValueError("api=True requires at least one api_key.")

        self.model = model
        self.api_base = api_base
        self.api_keys = keys
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.timeout = timeout
        self._client_cache = {}
        self._lock = threading.Lock()
        self._key_index = 0

    def generate(self, prompt: str) -> str:
        last_error = None
        for _ in range(self.max_retries):
            client = self._next_client()
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                )
                message = completion.choices[0].message.content
                return (message or "").strip()
            except Exception as exc:
                last_error = exc
                time.sleep(self.retry_sleep)
        raise RuntimeError(f"API generation failed after {self.max_retries} attempts: {last_error}")

    def _next_client(self):
        with self._lock:
            api_key = self.api_keys[self._key_index % len(self.api_keys)]
            self._key_index += 1
            client = self._client_cache.get(api_key)
            if client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError(
                        "API mode requires the 'openai' package to be installed."
                    ) from exc
                client = OpenAI(
                    base_url=self.api_base,
                    api_key=api_key,
                    max_retries=2,
                    timeout=self.timeout,
                )
                self._client_cache[api_key] = client
            return client


class LocalVLLMBackend(BaseLLMBackend):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        top_p: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self._llm = None
        self._sampling_params_cls = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def generate(self, prompt: str) -> str:
        self._ensure_model()
        formatted_prompt = self._format_prompt(prompt)
        outputs = self._llm.generate(
            [formatted_prompt],
            self._sampling_params_cls(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
            ),
        )
        return outputs[0].outputs[0].text.strip()

    def _ensure_model(self):
        with self._lock:
            if self._llm is not None:
                return
            try:
                from transformers import AutoTokenizer
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise RuntimeError(
                    "Local vLLM mode requires the 'vllm' and 'transformers' packages."
                ) from exc

            self._tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            self._llm = LLM(
                model=self.model,
                trust_remote_code=True,
                dtype="float16",
                tensor_parallel_size=1,
            )
            self._sampling_params_cls = SamplingParams

    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt


def build_llm_backend(
    *,
    api: bool,
    model: str,
    api_base: str = "",
    api_key=None,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
):
    if api:
        return OpenAICompatibleBackend(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    return LocalVLLMBackend(
        model=model,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def _normalize_api_keys(api_key) -> List[str]:
    if api_key is None:
        return []
    if isinstance(api_key, str):
        key = api_key.strip()
        return [key] if key else []
    if isinstance(api_key, Sequence):
        return [str(key).strip() for key in api_key if str(key).strip()]
    raise ValueError("api_key must be a string or a list of strings.")
