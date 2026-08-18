"""One function, `generate()`, over the configured provider (Gemini or Groq)."""
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings

# USD per 1M tokens (input, output): the cost axis of the Pareto frontier.
# Models missing from this table are costed at $0, so add your model's current price here.
PRICES = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


class LLMError(RuntimeError):
    pass


@dataclass
class Generation:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        price_in, price_out = PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6


@lru_cache
def _client():
    s = get_settings()
    if s.llm_provider == "gemini":
        if not s.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        from google import genai
        return genai.Client(api_key=s.gemini_api_key)
    if not s.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set")
    from groq import Groq
    return Groq(api_key=s.groq_api_key)


def generate(prompt: str, system: str = "") -> Generation:
    s = get_settings()
    client = _client()
    try:
        if s.llm_provider == "gemini":
            from google.genai import types
            resp = client.models.generate_content(
                model=s.gemini_model, contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system or None,
                                                   temperature=s.llm_temperature),
            )
            usage = resp.usage_metadata
            return Generation(resp.text or "", s.gemini_model,
                              (usage and usage.prompt_token_count) or 0,
                              (usage and usage.candidates_token_count) or 0)

        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(model=s.groq_model, messages=messages,
                                              temperature=s.llm_temperature)
        return Generation(resp.choices[0].message.content or "", s.groq_model,
                          resp.usage.prompt_tokens, resp.usage.completion_tokens)
    except Exception as e:
        raise LLMError(f"{s.llm_provider} call failed: {e}") from e
