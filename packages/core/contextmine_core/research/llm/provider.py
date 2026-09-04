"""LLM Provider abstraction using LangChain.

This module provides a unified interface for LLM interactions in the research agent,
supporting multiple providers (Anthropic, OpenAI) with structured output validation.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from contextmine_core.model_policy import ensure_model_calls_enabled
from contextmine_core.research.llm.structured import (
    coerce_json_string_fields,
    repair_structured_payload,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Provides a unified interface for:
    - Text generation with system prompts
    - Structured JSON output with Pydantic validation
    - Retry logic with exponential backoff
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name being used."""
        ...

    @property
    def provider_name(self) -> str:
        """Get the provider identity used for telemetry and persisted metadata."""
        return "unknown"

    def bind_tools(self, tools: Sequence[Any]) -> BaseChatModel:
        """Bind tools through the provider's public interface."""
        raise NotImplementedError(f"{type(self).__name__} does not support tool binding")

    @abstractmethod
    async def generate_text(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Generate text response.

        Args:
            system: System prompt
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Generated text response
        """
        ...

    @abstractmethod
    async def generate_structured[T: BaseModel](
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        output_schema: type[T],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured output validated against a Pydantic schema.

        Args:
            system: System prompt
            messages: List of message dicts with 'role' and 'content' keys
            output_schema: Pydantic model class for output validation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If output doesn't match schema after retries
        """
        ...


class LangChainProvider(LLMProvider):
    """LLM Provider implementation using LangChain.

    Supports Anthropic and OpenAI models with:
    - Automatic retry with exponential backoff and jitter
    - Structured output via with_structured_output()
    - Fallback to JSON parsing if structured output fails
    """

    def __init__(
        self,
        model: BaseChatModel,
        model_name: str,
        provider_name: str = "unknown",
        max_retries: int = 3,
    ):
        """Initialize the provider.

        Args:
            model: LangChain chat model instance
            model_name: Name of the model for logging
            max_retries: Maximum retry attempts for API calls
        """
        self._model = model
        self._model_name = model_name
        self._provider_name = provider_name
        self._max_retries = max_retries

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    def bind_tools(self, tools: Sequence[Any]) -> BaseChatModel:
        """Bind tools without exposing the underlying LangChain model."""
        return self._model.bind_tools(tools)

    def _build_messages(
        self,
        system: str,
        messages: Sequence[dict[str, str]],
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        """Convert message dicts to LangChain message objects."""
        result: list[SystemMessage | HumanMessage | AIMessage] = [SystemMessage(content=system)]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                result.append(HumanMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            elif role == "system":
                result.append(SystemMessage(content=content))
            else:
                # Default to human message for unknown roles
                result.append(HumanMessage(content=content))

        return result

    def _generation_parameters(self, *, max_tokens: int, temperature: float) -> dict[str, object]:
        """Build supported generation parameters from the model capability profile."""
        parameters: dict[str, object] = {"max_tokens": max_tokens}
        profile = getattr(self._model, "profile", None)
        if not (isinstance(profile, dict) and profile.get("temperature") is False):
            parameters["temperature"] = temperature
        return parameters

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate_text(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Generate text response with retry logic."""
        ensure_model_calls_enabled()
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        tracer = trace.get_tracer(__name__)

        generation_parameters = self._generation_parameters(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        span_attributes: dict[str, object] = {
            "gen_ai.system": self._provider_name,
            "gen_ai.request.model": self._model_name,
            "gen_ai.operation.name": "chat",
            "gen_ai.request.max_tokens": max_tokens,
        }
        if "temperature" in generation_parameters:
            span_attributes["gen_ai.request.temperature"] = temperature
        with tracer.start_as_current_span(
            "llm.generate_text",
            attributes=span_attributes,
        ) as span:
            try:
                lc_messages = self._build_messages(system, messages)

                # Configure model with parameters
                configured_model = self._model.bind(**generation_parameters)

                response = await configured_model.ainvoke(lc_messages)

                # Record usage metrics if available
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = response.usage_metadata
                    if hasattr(usage, "input_tokens"):
                        span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
                    if hasattr(usage, "output_tokens"):
                        span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)

                return str(response.content)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate_structured[T: BaseModel](
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        output_schema: type[T],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured output with validation and retry."""
        ensure_model_calls_enabled()
        lc_messages = self._build_messages(system, messages)

        # Try using LangChain's built-in structured output first
        try:
            # Use default method - Anthropic uses tool calling, OpenAI uses json_schema.
            # include_raw keeps the tool-call arguments available so a response
            # that only fails schema validation can still be repaired below.
            structured_model = self._model.with_structured_output(output_schema, include_raw=True)
            configured_model = structured_model.bind(
                **self._generation_parameters(
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            )
            result = await configured_model.ainvoke(lc_messages)

            # Ensure we got the right type
            if isinstance(result, output_schema):
                return result

            if isinstance(result, dict) and "parsed" in result:
                parsed = result["parsed"]
                if isinstance(parsed, output_schema):
                    return parsed
                repaired = repair_structured_payload(result.get("raw"), output_schema)
                if repaired is not None:
                    return repaired
                parsing_error = result.get("parsing_error")
                if parsing_error is not None:
                    raise parsing_error
                msg = f"Model returned no usable {output_schema.__name__}"
                raise ValueError(msg)

            # If we got a dict, try to parse it
            if isinstance(result, dict):
                return output_schema.model_validate(
                    coerce_json_string_fields(result, output_schema)
                )

            # Raise a ValueError that will be caught and retried
            msg = f"Expected {output_schema.__name__}, got {type(result).__name__}"
            raise ValueError(msg)

        except NotImplementedError:
            # Fallback: Use output parser if structured output not supported
            logger.debug(
                "Structured output not supported for %s, using parser fallback",
                self._model_name,
            )
            return await self._generate_with_parser(
                system=system,
                messages=messages,
                output_schema=output_schema,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    async def _generate_with_parser[T: BaseModel](
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        output_schema: type[T],
        max_tokens: int,
        temperature: float,
    ) -> T:
        """Fallback: Generate text and parse with Pydantic."""
        parser = PydanticOutputParser(pydantic_object=output_schema)
        format_instructions = parser.get_format_instructions()

        # Append format instructions to the last user message
        enhanced_messages = list(messages)
        if enhanced_messages:
            last_msg = enhanced_messages[-1].copy()
            last_msg["content"] = f"{last_msg['content']}\n\n{format_instructions}"
            enhanced_messages[-1] = last_msg
        else:
            enhanced_messages.append({"role": "user", "content": format_instructions})

        text_response = await self.generate_text(
            system=system,
            messages=enhanced_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return parser.parse(text_response)


def get_llm_provider(
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    timeout: float = 60.0,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider: Provider name ('anthropic', 'openai')
        model: Model name (defaults to provider's default)
        api_key: API key (defaults to environment variable)
        max_retries: Maximum retry attempts
        timeout: Request timeout in seconds

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    ensure_model_calls_enabled()
    from contextmine_core.settings import get_settings

    settings = get_settings()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        selected_model = model or settings.anthropic_llm_model
        # Read from environment if not provided
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        # ChatAnthropic and ChatOpenAI accept the model id under different kwarg names
        # in their type stubs (model_name vs model); keep them asymmetric so ty passes.
        chat_model = ChatAnthropic(
            model_name=selected_model,
            api_key=resolved_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        return LangChainProvider(
            model=chat_model,
            model_name=selected_model,
            provider_name="anthropic",
            max_retries=max_retries,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        selected_model = model or settings.openai_llm_model
        # Read from environment if not provided
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        chat_model = ChatOpenAI(
            model=selected_model,
            api_key=resolved_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        return LangChainProvider(
            model=chat_model,
            model_name=selected_model,
            provider_name="openai",
            max_retries=max_retries,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'anthropic' or 'openai'.")


def get_research_llm_provider() -> LLMProvider:
    """Get the LLM provider configured for research agent use.

    Uses settings from contextmine_core.settings:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY
    - RESEARCH_MODEL
    - RESEARCH_MAX_TOKENS

    Returns:
        Configured LLMProvider for research agent
    """
    ensure_model_calls_enabled()
    from contextmine_core.settings import get_settings, secret_value

    settings = get_settings()

    # Determine provider based on available API keys and model name
    model = settings.research_model

    if model.startswith("claude") or model.startswith("anthropic"):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for Claude models")
        return get_llm_provider(
            provider="anthropic",
            model=model,
            api_key=secret_value(settings.anthropic_api_key),
        )
    elif model.startswith("gpt") or model.startswith("o1"):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI models")
        return get_llm_provider(
            provider="openai",
            model=model,
            api_key=secret_value(settings.openai_api_key),
        )
    else:
        # Default to Anthropic if key available
        if settings.anthropic_api_key:
            return get_llm_provider(
                provider="anthropic",
                model=model,
                api_key=secret_value(settings.anthropic_api_key),
            )
        elif settings.openai_api_key:
            return get_llm_provider(
                provider="openai",
                model=model,
                api_key=secret_value(settings.openai_api_key),
            )
        else:
            raise ValueError("No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
