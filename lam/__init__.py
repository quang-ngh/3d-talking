"""
LAM Refactored - Modular Architecture for Large Avatar Model

A clean, modular refactoring of the LAM codebase with clear component boundaries.
"""

__version__ = "0.1.0"

# Core components
from . import flame
from . import gaussian
from . import utils

# Optional components (may require additional dependencies)
try:
    from . import tracking
except ImportError:
    tracking = None

try:
    from . import encoder
except ImportError:
    encoder = None

try:
    from . import model
except ImportError:
    model = None

try:
    from . import pipeline
except ImportError:
    pipeline = None

__all__ = [
    "flame",
    "gaussian",
    "utils",
    "tracking",
    "encoder",
    "model",
    "pipeline",
]
