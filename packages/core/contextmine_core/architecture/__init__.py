"""Architecture facts, arc42 generation, and drift reporting."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "SECTION_TITLES",
    "ArchitectureClaim",
    "Arc42ClaimTraceability",
    "ClaudeAgentSdkUnavailableError",
    "Arc42Document",
    "Arc42DriftReport",
    "ArchitectureFact",
    "ArchitectureFactsBundle",
    "DriftDelta",
    "EvidenceRef",
    "PortAdapterFact",
    "build_architecture_facts",
    "claim_counter_evidence",
    "claim_supporting_evidence",
    "compute_arc42_drift",
    "generate_arc42_from_facts",
    "generate_arc42_with_claude_sdk",
    "normalize_arc42_section_key",
    "summarize_confidence",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "ClaudeAgentSdkUnavailableError": (".agent_sdk", "ClaudeAgentSdkUnavailableError"),
    "generate_arc42_with_claude_sdk": (".agent_sdk", "generate_arc42_with_claude_sdk"),
    "SECTION_TITLES": (".arc42", "SECTION_TITLES"),
    "generate_arc42_from_facts": (".arc42", "generate_arc42_from_facts"),
    "normalize_arc42_section_key": (".arc42", "normalize_arc42_section_key"),
    "Arc42ClaimTraceability": (".claim_model", "Arc42ClaimTraceability"),
    "ArchitectureClaim": (".claim_model", "ArchitectureClaim"),
    "claim_counter_evidence": (".claim_model", "claim_counter_evidence"),
    "claim_supporting_evidence": (".claim_model", "claim_supporting_evidence"),
    "compute_arc42_drift": (".drift", "compute_arc42_drift"),
    "build_architecture_facts": (".facts", "build_architecture_facts"),
    "Arc42Document": (".schemas", "Arc42Document"),
    "Arc42DriftReport": (".schemas", "Arc42DriftReport"),
    "ArchitectureFact": (".schemas", "ArchitectureFact"),
    "ArchitectureFactsBundle": (".schemas", "ArchitectureFactsBundle"),
    "DriftDelta": (".schemas", "DriftDelta"),
    "EvidenceRef": (".schemas", "EvidenceRef"),
    "PortAdapterFact": (".schemas", "PortAdapterFact"),
    "summarize_confidence": (".schemas", "summarize_confidence"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
