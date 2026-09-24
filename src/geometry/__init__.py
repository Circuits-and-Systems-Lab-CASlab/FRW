"""Format-independent geometry models shared by BGS, FRW, and input adapters."""

from .models import Bounds, Conductor, DIRECTIONS, DielectricRegion, GeometryModel

__all__ = [
    "Bounds",
    "Conductor",
    "DIRECTIONS",
    "DielectricRegion",
    "GeometryModel",
]
