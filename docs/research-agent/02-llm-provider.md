# LLM Provider Abstraction

This document describes the LLM provider abstraction layer for the research agent,
implemented as part of Prompt 2.

## Overview

The LLM provider layer provides a unified interface for interacting with language models
(Anthropic Claude, OpenAI GPT) using LangChain as the underlying framework.

Key features:
- **Unified interface** via `LLMProvider` abstract base class
- **Structured output** with Pydantic schema validation
- **Retry with exponential backoff** using tenacity
- **Prompt injection firewall** for security
- **Mock providers** for deterministic testing

## Architecture

```
contextmine_core/research/llm/
├── __init__.py          # Package exports
├── provider.py          # LLMProvider ABC + LangChainProvider implementation
├── prompts.py           # Prompt templates and security policy
└── mock.py              # Mock providers for testing
```

## LLMProvider Interface

The `LLMProvider` abstract base class defines two core methods:

```python
from contextmine_core.research.llm import LLMProvider

class LLMProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name being used."""
        ...

    @abstractmethod
    async def generate_text(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Generate text response."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        *,
        system: str,
        messages: Sequence[dict[str, str]],
        output_schema: type[T],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured output validated against a Pydantic schema."""
        ...
```

## Usage

### Creating a Provider

Use the factory functions to create providers:

```python
from contextmine_core.research.llm import get_llm_provider, get_research_llm_provider

# Explicit provider creation
provider = get_llm_provider(
    provider="anthropic",  # or "openai"
    model="claude-sonnet-4-20250514",
    api_key="your-api-key",
    max_retries=3,
    timeout=60.0,
)

# Or use settings-based creation (reads from environment)
provider = get_research_llm_provider()
```

### Text Generation

```python
response = await provider.generate_text(
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100,
    temperature=0.0,
)
print(response)  # "Paris is the capital of France."
```

### Structured Output

```python
from pydantic import BaseModel

class SearchAction(BaseModel):
    action: str
    query: str
    reasoning: str

result = await provider.generate_structured(
    system="You are a code search assistant.",
    messages=[
        {"role": "user", "content": "Find authentication code"}
    ],
    output_schema=SearchAction,
)
print(result.action)    # "hybrid_search"
print(result.query)     # "authentication login"
print(result.reasoning) # "Need to find auth-related code"
```

## Prompt Injection Firewall

All research agent prompts include a security firewall policy that:

1. **Treats repository content as untrusted data**
2. **Uses code only as evidence** for answering questions
3. **Ignores manipulation attempts** in code/comments
4. **Sticks to the research task** only

```python
from contextmine_core.research.llm import build_research_system_prompt

prompt = build_research_system_prompt(
    question="How does authentication work?",
    scope="src/auth/**",  # Optional: limit search scope
    additional_context="Focus on OAuth implementation",  # Optional
)
```

The firewall policy is prepended to all research agent system prompts:

```python
from contextmine_core.research.llm import PROMPT_INJECTION_FIREWALL
# Contains critical security instructions
```

## Testing

### MockLLMProvider

For deterministic testing without API calls:

```python
from contextmine_core.research.llm import MockLLMProvider

provider = MockLLMProvider(
    model_name="test-model",
    default_text_response="Mocked response",
)

# Configure specific structured responses
provider.set_structured_response("SearchAction", {
    "action": "hybrid_search",
    "query": "test",
    "reasoning": "testing",
})

# Inspect call history
result = await provider.generate_text(...)
assert len(provider.call_history) == 1
assert provider.call_history[0]["method"] == "generate_text"

# Reset for next test
provider.reset_history()
```

### FailingMockProvider

For testing retry behavior:

```python
from contextmine_core.research.llm import FailingMockProvider

provider = FailingMockProvider(
    fail_count=2,  # Fail first 2 calls
    error_type=ConnectionError,
)

# First two calls raise ConnectionError
# Third call succeeds
```

## Configuration

Settings are loaded from `contextmine_core.settings`:

| Setting | Environment Variable | Default |
|---------|---------------------|---------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | None |
| `openai_api_key` | `OPENAI_API_KEY` | None |
| `research_model` | `RESEARCH_MODEL` | `claude-sonnet-4-20250514` |

## Dependencies

The LLM provider uses LangChain 1.x:

```toml
[project.dependencies]
langchain-core = ">=1.0.0"
langchain-anthropic = ">=1.0.0"
langchain-openai = ">=1.0.0"
langgraph = ">=1.0.0"
tenacity = ">=8.0.0"
```

## Error Handling

The provider includes automatic retry with exponential backoff and jitter:

- **Retried errors**: `ConnectionError`, `TimeoutError`, `ValidationError`, `ValueError`
- **Backoff**: Exponential with jitter (1-10 seconds)
- **Max attempts**: 3 (configurable)

```python
@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, ValidationError, ValueError)),
    wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def generate_structured(...):
    ...
```

## Next Steps

The LLM provider is used by:

- **Prompt 3**: Action schemas and agent loop (LangGraph)
- **Prompt 4**: Tool implementations (search, read spans)
- **Prompt 5**: MCP tool integration (`context.research`)
