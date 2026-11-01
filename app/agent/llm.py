"""Universal LLM provider abstraction backed by LiteLLM.

Agents only ever see one object with an ``invoke(messages)`` method that returns
a LangChain ``AIMessage``, so switching providers is configuration, not code.

Configuration (``.env``):

  ``LLM_PROVIDER``     openai | gemini | perplexity | groq | <any LiteLLM prefix>
  ``LLM_MODEL``        optional fully-qualified override, e.g. ``groq/llama-3.3-70b-versatile``
  ``LLM_API_KEY``      optional generic key; otherwise the provider's own env var is used
  ``LLM_TEMPERATURE``  sampling temperature (default 0.7)
  ``LLM_TIMEOUT_SECONDS`` request timeout (default 90)
  ``LLM_MAX_RETRIES``  SDK-level retries (default 0; the job queue retries instead)

Built-in providers and their keys:

  * **openai**     — ``OPENAI_API_KEY``,     default ``gpt-4o-mini``
  * **gemini**     — ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``, default ``gemini-2.5-flash``
  * **perplexity** — ``PERPLEXITY_API_KEY``, default ``sonar-pro``
  * **groq**       — ``GROQ_API_KEY``,       default ``llama-3.3-70b-versatile``

Any other LiteLLM provider works without touching this file: set
``LLM_PROVIDER=anthropic`` and ``LLM_MODEL=claude-sonnet-4-5`` (its native
``ANTHROPIC_API_KEY`` is picked up by LiteLLM automatically).

Note: ``litellm`` is imported lazily inside ``invoke`` so importing the agent
package stays cheap and the pinned LangChain stack is untouched at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from langchain_core.messages import AIMessage

#: LangChain message ``type`` -> OpenAI/LiteLLM chat role.
_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "ai": "assistant",
    "tool": "tool",
    "function": "tool",
}


@dataclass(frozen=True)
class _Provider:
    """Everything needed to turn ``LLM_PROVIDER`` into a LiteLLM call."""

    prefix: str
    api_key_envs: tuple[str, ...]
    model_env: str
    default_model: str


_PROVIDERS: dict[str, _Provider] = {
    "openai": _Provider(
        prefix="openai",
        api_key_envs=("OPENAI_API_KEY",),
        model_env="OPENAI_MODEL",
        default_model="gpt-4o-mini",
    ),
    "gemini": _Provider(
        prefix="gemini",
        # LiteLLM reads GEMINI_API_KEY; the project historically used
        # GOOGLE_API_KEY, so both are accepted.
        api_key_envs=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        model_env="GEMINI_MODEL",
        default_model="gemini-2.5-flash",
    ),
    "perplexity": _Provider(
        prefix="perplexity",
        api_key_envs=("PERPLEXITY_API_KEY",),
        model_env="PERPLEXITY_MODEL",
        default_model="sonar-pro",
    ),
    "groq": _Provider(
        prefix="groq",
        api_key_envs=("GROQ_API_KEY",),
        model_env="GROQ_MODEL",
        default_model="llama-3.3-70b-versatile",
    ),
}

#: Friendly aliases that should not require a separate provider block.
_ALIASES = {
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "open-ai": "openai",
}


def _get_env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _get_float_env(key: str, default: float) -> float:
    raw = _get_env(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    raw = _get_env(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _provider_name(raw: str) -> str:
    name = (raw or "").strip().lower()
    return _ALIASES.get(name, name)


def _resolve_model(provider: str, model: str | None) -> tuple[str, _Provider | None]:
    """Return ``(litellm_model, spec)`` with a provider prefix applied once."""
    spec = _PROVIDERS.get(provider)
    explicit = (model or "").strip()
    generic = _get_env("LLM_MODEL")

    if explicit:
        chosen = explicit
    elif generic:
        chosen = generic
    elif spec:
        chosen = _get_env(spec.model_env) or spec.default_model
    else:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Set LLM_MODEL to a fully-qualified "
            f"LiteLLM model (e.g. 'anthropic/claude-sonnet-4-5'), or use one of: "
            f"{', '.join(sorted(_PROVIDERS))}."
        )

    if "/" not in chosen:
        chosen = f"{spec.prefix if spec else provider}/{chosen}"
    return chosen, spec


def _resolve_api_key(provider: str, spec: _Provider | None) -> str:
    """Key precedence: LLM_API_KEY, the provider's own vars, then ``""``.

    An empty key is fine: LiteLLM falls back to the native environment variable
    for whichever provider the model string names (including ones we don't
    list here), so adding a provider never requires a code change.
    """
    generic = _get_env("LLM_API_KEY")
    if generic:
        return generic
    envs = spec.api_key_envs if spec else (f"{provider.upper().replace('-', '_')}_API_KEY",)
    for env in envs:
        value = _get_env(env)
        if value:
            return value
    return ""


def _content_to_text(content: Any) -> str:
    """Flatten LiteLLM content (str or multimodal parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _to_chat_message(message: Any) -> dict[str, Any]:
    """Convert a LangChain message (or dict) into LiteLLM's chat format."""
    if isinstance(message, dict):
        return message
    role = _ROLE_MAP.get(getattr(message, "type", ""), "user")
    return {"role": role, "content": _content_to_text(getattr(message, "content", ""))}


def _usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    else:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
    return int(prompt), int(completion)


class LiteLLMChatModel:
    """The small LangChain surface StayScout needs, implemented over LiteLLM.

    ``runs.tracked_invoke`` reads ``model_name``, calls ``invoke`` and then pulls
    ``content`` / ``usage_metadata`` off the response, so returning an
    ``AIMessage`` keeps the cost ledger working unchanged.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str,
        temperature: float = 0.7,
        api_key: str = "",
        timeout: float = 90.0,
        max_retries: int = 0,
    ) -> None:
        self.model = model
        self.model_name = model
        self.provider = provider
        self.temperature = temperature
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries

    def invoke(self, messages: Sequence[Any], **kwargs: Any) -> AIMessage:
        """Call the configured model and return a LangChain ``AIMessage``."""
        try:
            from litellm import completion
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ImportError(
                "litellm is not installed. Run `pip install -r requirements.txt` "
                "or `uv pip install litellm`."
            ) from exc

        params: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_chat_message(m) for m in messages],
            "temperature": self.temperature,
            "timeout": self._timeout,
        }
        if self._api_key:
            params["api_key"] = self._api_key
        if self._max_retries > 0:
            params["num_retries"] = self._max_retries
        params.update(kwargs)

        response = completion(**params)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        text = _content_to_text(getattr(message, "content", "")).strip()
        prompt_tokens, completion_tokens = _usage_tokens(response)

        return AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            response_metadata={
                "provider": self.provider,
                "model": self.model,
                "finish_reason": getattr(choice, "finish_reason", None),
            },
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LiteLLMChatModel(model={self.model!r}, provider={self.provider!r})"


@lru_cache(maxsize=1)
def get_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> LiteLLMChatModel:
    """Return a chat model for the configured (or given) provider.

    The result is cached because every agent calls this without arguments and
    the configuration does not change while a worker is running; call
    ``get_llm.cache_clear()`` after changing the environment.
    """
    raw_provider = (provider or _get_env("LLM_PROVIDER")).strip()
    name = _provider_name(raw_provider) if raw_provider else ""
    if not name:
        # No explicit provider: a namespaced LLM_MODEL identifies it, otherwise
        # keep the historical OpenAI default.
        generic = _get_env("LLM_MODEL")
        name = _provider_name(generic.split("/", 1)[0]) if "/" in generic else "openai"

    chosen_model, spec = _resolve_model(name, model)
    temp = (
        temperature
        if temperature is not None
        else _get_float_env("LLM_TEMPERATURE", 0.7)
    )
    return LiteLLMChatModel(
        model=chosen_model,
        provider=name,
        temperature=temp,
        api_key=_resolve_api_key(name, spec),
        timeout=_get_float_env("LLM_TIMEOUT_SECONDS", 90.0),
        max_retries=_get_int_env("LLM_MAX_RETRIES", 0),
    )
