# ORBIT: Language-Invariant Multimodal Bi-Geometric Adversarial Learning for Zero-Shot Cross-Lingual Alzheimer's Detection

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Interspeech%202026-blue)](https://www.isca-archive.org/interspeech_2026/girish26b_interspeech.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)

Zero-shot cross-lingual speech-based Alzheimer's detection using multimodal fusion and adversarial learning

</div>

## Overview

**ORBIT** is a framework for detecting Alzheimer's disease from speech in languages not seen during training. Unlike traditional models that learn language-specific patterns, ORBIT learns **language-invariant multimodal representations** by fusing speech and text from multilingual pretrained models.

### Key Problem

Speech-based Alzheimer's detection models fail on unseen languages because they pick up language-specific patterns (accent, grammar, dialect) rather than signs of cognitive impairment. ORBIT solves this through adversarial training and geometric learning.

### Key Solution

**Synergizing three components:**

1. **Multimodal Fusion** - Combines acoustic markers (speech) with linguistic markers (text)
2. **Adversarial Learning** - Multi-tap language adversaries suppress language-specific information
3. **Bi-Geometric Learning** - Spherical-hyperbolic projections with consensus clustering

### Results

**Zero-shot evaluation** (trained on English, tested on unseen languages):
- **Consistently outperforms** unimodal baselines by 6-9%
- **Beats concatenation fusion** through learned cross-attention  
- **State-of-the-art** on zero-shot cross-lingual Alzheimer's detection

## Paper

**Synergizing Zero-Shot Cross-Lingual Alzheimer Detection with Language-Invariant Multimodal Bi-Geometric Adversarial Learning**

- **Authors**: Girish, Mohd Mujtaba Akhtar, Farhan Sheth, Muskaan Singh, Juliana Gerard, Paula McClean, Kongfatt Wong-Lin
- **Conference**: Interspeech 2026
- **Paper Link**: [ISCA Archive](https://www.isca-archive.org/interspeech_2026/girish26b_interspeech.html)

## Quick Start

```bash
git clone https://github.com/Helixometry/ORBIT.git
cd ORBIT
pip install -r requirements.txt
python scripts/train.py --config configs/config_multimodal.yaml
```

## Key Components

1. **Multimodal Fusion** - Cross-attention fusion of speech and text
2. **Language Adversaries** - Multi-tap discriminators suppress language-specific patterns
3. **Bi-Geometric Learning** - Spherical + Hyperbolic projections

## Citation

```bibtex
@inproceedings{girish2026orbit,
  title={Synergizing Zero-Shot Cross-Lingual Alzheimer Detection with Language-Invariant Multimodal Bi-Geometric Adversarial Learning},
  author={Girish and Akhtar, Mohd Mujtaba and Sheth, Farhan and Singh, Muskaan and Gerard, Juliana and McClean, Paula and Wong-Lin, Kongfatt},
  booktitle={Proceedings of Interspeech 2026},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.