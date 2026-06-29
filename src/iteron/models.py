import os
import time
from typing import Optional


class ModelError(Exception):
    pass


class ModelClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.environ.get("ITERON_MODEL_PROVIDER", "anthropic")
        self._fast_model = os.environ.get("ITERON_FAST_MODEL", "")
        self._smart_model = os.environ.get("ITERON_SMART_MODEL", "")
        self._client = None
        self._base_url = None
        self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"]
            )
            if not self._fast_model:
                self._fast_model = "claude-3-5-haiku-latest"
            if not self._smart_model:
                self._smart_model = "claude-3-opus-latest"
        elif self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            if not self._fast_model:
                self._fast_model = "gpt-4o-mini"
            if not self._smart_model:
                self._smart_model = "gpt-4o"
        elif self.provider == "ollama":
            self._base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            if not self._fast_model:
                self._fast_model = "llama3.2"
            if not self._smart_model:
                self._smart_model = "llama3.2"
        else:
            raise ModelError(f"Unknown provider: {self.provider}")

    def call(
        self,
        prompt: str,
        system: str = "",
        tier: str = "fast",
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> str:
        model = self._fast_model if tier == "fast" else self._smart_model
        last_error = None
        for attempt in range(max_retries):
            try:
                return self._call_once(model, prompt, system, temperature)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        raise ModelError(
            f"Model call failed after {max_retries} retries: {last_error}"
        )

    def _call_once(
        self, model: str, prompt: str, system: str, temperature: float
    ) -> str:
        if self.provider == "anthropic":
            kwargs = dict(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if system:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return resp.content[0].text
        elif self.provider == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        elif self.provider == "ollama":
            import httpx
            payload = dict(
                model=model,
                system=system,
                prompt=prompt,
                stream=False,
                temperature=temperature,
            )
            resp = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        raise ModelError(f"Unhandled provider: {self.provider}")

    def estimate_cost(self, text: str, tier: str) -> float:
        # ponytail: rough token estimate (1.3x word count); tiktoken if precision matters
        tokens = len(text.split()) * 1.3
        rate = 0.001 if tier == "fast" else 0.075
        return tokens * rate / 1000

    @property
    def fast_model(self) -> str:
        return self._fast_model

    @property
    def smart_model(self) -> str:
        return self._smart_model
