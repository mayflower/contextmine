"""Architecture facts, arc42 generation, and drift reporting."""

from .arc42 import SECTION_TITLES, generate_arc42_from_facts, normalize_arc42_section_key
from .drift import compute_arc42_drift
from .facts import build_architecture_facts
from .schemas import (
    Arc42Document,
    Arc42DriftReport,
    ArchitectureFact,
    ArchitectureFactsBundle,
    DriftDelta,
    EvidenceRef,
    PortAdapterFact,
    summarize_confidence,
)

__all__ = [
    "SECTION_TITLES",
    "Arc42Document",
    "Arc42DriftReport",
    "ArchitectureFact",
    "ArchitectureFactsBundle",
    "DriftDelta",
    "EvidenceRef",
    "PortAdapterFact",
    "build_architecture_facts",
    "compute_arc42_drift",
    "generate_arc42_from_facts",
    "normalize_arc42_section_key",
    "summarize_confidence",
]
