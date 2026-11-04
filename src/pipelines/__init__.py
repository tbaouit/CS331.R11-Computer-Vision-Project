"""High-level Amodal Appearance Completion pipeline orchestrators and runtime dispatcher."""
from src.pipelines.backend import SCAmodalBackend
from src.pipelines.improved_completer import ImprovedAmodalCompleter
from src.pipelines.oracle_completer import OracleMaskAmodalCompleter

__all__ = [
    "ImprovedAmodalCompleter",
    "OracleMaskAmodalCompleter",
    "SCAmodalBackend",
]
