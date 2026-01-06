from __future__ import annotations

import os
from typing import Dict

from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM calls.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "600")),
        )
        if not resp.choices:
            raise RuntimeError("No choices returned from LLM")
        return resp.choices[0].message.content or ""
