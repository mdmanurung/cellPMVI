"""cellPMVI: Multi-modal VAE for single-cell RNA + protein data."""

from cellpmvi.model._cellpmvi import CellPMVI
from cellpmvi.model._cellpmvi_citeseq import CellPMVICiteseq

__all__ = ["CellPMVI", "CellPMVICiteseq"]
__version__ = "0.1.0"
