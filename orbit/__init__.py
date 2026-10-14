"""ORBIT: Language-Invariant Multimodal Bi-Geometric Adversarial Learning for Zero-Shot Cross-Lingual Alzheimer's Detection"""

__version__ = "1.0.0"
__author__ = "Girish, Mohd Mujtaba Akhtar, Farhan Sheth, Muskaan Singh, Juliana Gerard, Paula McClean, Kongfatt Wong-Lin"
__email__ = "girish281003@gmail.com"

from .model import ORBIT
from .fusion import MultimodalFusion
from .adversary import LanguageAdversary
from .geometry import BiGeometricProjection

__all__ = [
    "ORBIT",
    "MultimodalFusion",
    "LanguageAdversary",
    "BiGeometricProjection",
]
