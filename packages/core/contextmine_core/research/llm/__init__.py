"""LLM provider abstraction using LangChain for research agent."""

from contextmine_core.research.llm.mock import FailingMockProvider, MockLLMProvider
from contextmine_core.research.llm.prompts import (
    PROMPT_INJECTION_FIREWALL,
    build_action_selection_prompt,
    build_research_system_prompt,
)
from contextmine_core.research.llm.provider import (
    LangChainProvider,
    LLMProvider,
    get_llm_provider,
    get_research_llm_provider,
)

__all__ = [
    "FailingMockProvider",
    "LLMProvider",
    "LangChainProvider",
    "MockLLMProvider",
    "PROMPT_INJECTION_FIREWALL",
    "build_action_selection_prompt",
    "build_research_system_prompt",
    "get_llm_provider",
    "get_research_llm_provider",
]
