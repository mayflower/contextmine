"""LLM-based business rule extraction.

This module is kept for backwards compatibility.
The actual extraction now happens in extractors/rules.py using LLM directly.

The previous two-step approach (structural extraction → LLM labeling) has been
replaced with a single LLM-based extraction that:
- Works with all 12 Tree-sitter languages
- Handles any human language (German, Spanish, Dutch, etc.)
- Detects any pattern (decorators, annotations, schemas, guards, etc.)
"""

from __future__ import annotations

# Re-export from new location for backwards compatibility
from contextmine_core.analyzer.extractors.rules import (
    BusinessRuleDef,
    ExtractedRule,
    ExtractionOutput,
    RulesExtraction,
    build_rules_graph,
    extract_rules_from_file,
)
from contextmine_core.analyzer.extractors.rules import (
    ExtractionOutput as BusinessRuleOutput,
)

__all__ = [
    "BusinessRuleDef",
    "BusinessRuleOutput",
    "ExtractedRule",
    "ExtractionOutput",
    "RulesExtraction",
    "build_rules_graph",
    "extract_rules_from_file",
]
