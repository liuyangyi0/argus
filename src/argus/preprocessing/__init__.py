"""Frame preprocessing stages that run between capture and detection."""

from argus.preprocessing.alignment import PhaseCorrelator, create_from_config

__all__ = ["PhaseCorrelator", "create_from_config"]
