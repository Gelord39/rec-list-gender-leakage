"""LLM backends for list AIA.

- OllamaBackend: local (default for now)
- OpenAICompatibleBackend: placeholder for later API use (OpenAI / vLLM / etc.)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any] | None = None
    # Per generated token: {"token", "logprob", "top_logprobs": [{"token", "logprob"}, ...]}
    logprobs: list[dict[str, Any]] | None = None


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.0,
        timeout: int = 300,
        seed: int | None = None,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        logprobs: bool = True,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.seed = seed
        self.num_predict = num_predict
        self.num_ctx = num_ctx  # Ollama defaults to 4096; many-shot prompts need more
        self.logprobs = logprobs

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        url = self.host + "/api/generate"
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict  # max output tokens
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx  # context window; prompts above it are truncated
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if self.logprobs:
            # Does not change the generated text; lets list_aia derive a
            # continuous P(male) score at the answer token (for AUC).
            payload["logprobs"] = True
            payload["top_logprobs"] = 20  # Ollama's maximum
        if system:
            payload["system"] = system
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. "
                f"Run `ollama serve` and `ollama pull {self.model}`. ({exc})"
            ) from exc
        return LLMResponse(
            text=data.get("response", ""), raw=data, logprobs=data.get("logprobs")
        )


class OpenAICompatibleBackend(LLMBackend):
    """Stub / thin client for OpenAI-compatible chat APIs.

    Not used in the Ollama pilot; kept so switching later is one flag.
    Requires OPENAI_API_KEY in the environment or project-root ``.env``.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OpenAICompatibleBackend needs api_key or OPENAI_API_KEY")

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        url = self.base_url + "/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw=data)


def make_backend(
    kind: str,
    model: str,
    *,
    host: str = "http://127.0.0.1:11434",
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.0,
    num_ctx: int | None = None,
) -> LLMBackend:
    kind = kind.lower()
    if kind == "ollama":
        return OllamaBackend(model=model, host=host, temperature=temperature, num_ctx=num_ctx)
    if kind in {"openai", "api", "openai_compatible"}:
        return OpenAICompatibleBackend(
            model=model, api_key=api_key, base_url=base_url, temperature=temperature
        )
    raise ValueError(f"Unknown backend kind: {kind!r} (use ollama|openai)")
