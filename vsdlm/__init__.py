"""VSDLM: Visual-only speech detection by lip movement."""

from .model import VSDLM
from .pipeline import main

__all__ = ["VSDLM", "main"]
