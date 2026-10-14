# ORBIT Architecture

## Overview

ORBIT (Language-Invariant Multimodal Bi-Geometric Adversarial Learning) detects Alzheimer's disease in zero-shot cross-lingual settings by synergizing three key components.

## Three Core Components

### 1. Multimodal Fusion

**Cross-Attention Based Fusion**

- Speech Encoder: WavLM or UniSpeech (1024-dim)
- Text Encoder: XLM-R or mBERT (768-dim)
- Fusion: Learnable cross-attention mechanism

```
Speech (1024) ──┐
               ├─→ Cross-Attention ──→ Fused (512)
Text (768) ────┘      Fusion Layer
```

**Benefits:**
- Learns optimal weight for combining modalities
- Captures speech-text interactions
- Better than simple concatenation

### 2. Language Adversaries (Multi-Tap)

**Suppresses Language-Specific Information**

- Multiple discriminators at different representation levels
- Gradient reversal for adversarial training
- Prevents model from learning language-specific artifacts

```
Fused Features ──→ Adversary 1 (Level 1) ──→ Language Prediction
                ──→ Adversary 2 (Level 2) ──→ Language Prediction
                ──→ Adversary 3 (Level 3) ──→ Language Prediction
                        (Multi-Tap)
```

**Multi-Tap Approach:**
- Tap 1: Early fusion representations
- Tap 2: Mid-level abstract features
- Tap 3: Late high-level representations

### 3. Bi-Geometric Projections

**Spherical + Hyperbolic Spaces**

**Spherical Projection:**
- Normalizes representations to unit sphere
- Models angular distances
- Captures directional similarity

**Hyperbolic Projection:**
- Poincaré ball model
- Hierarchical structure
- Captures tree-like relationships in disease manifestations

**Consensus Clustering:**
- Aggregates information across spaces
- Leverages geometric properties

## Data Flow

```
┌─────────────────────────────────────────────────────┐
│             Input Speech + Transcription              │
└──────────┬──────────────────────┬───────────────────┘
           ↓                      ↓
┌──────────────────┐  ┌─────────────────────┐
│ Speech Encoder   │  │ Text Encoder        │
│ (WavLM/UniSpeech)│  │ (XLM-R/mBERT)       │
│ Output: 1024-dim │  │ Output: 768-dim     │
└────────┬─────────┘  └──────────┬──────────┘
         │                       │
         └───────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │  Projection Layers      │
        │  (1024→512, 768→512)   │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │ Cross-Attention Fusion │
        │      (2×512→512)       │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │ Language Adversaries    │
        │   (Multi-Tap: 3×)      │
        │ (Suppress lang-specific)
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │ Bi-Geometric Projection│
        │ (Spherical + Hyperbolic)
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │ Classification Head    │
        │    (Binary: HC/AD)     │
        └────────────┬───────────┘
                     ↓
        ┌────────────────────────┐
        │   Output Logits        │
        │  + Probabilities       │
        └────────────────────────┘
```

## Training Objective

```
L_total = L_task + λ_adv × L_adversary + λ_geo × L_geometry

L_task: Classification loss (AD detection)
L_adversary: Adversarial loss (suppress language info)
L_geometry: Geometric consistency loss
```

### Loss Components

1. **Task Loss**: Cross-entropy for AD vs Healthy classification
2. **Adversarial Loss**: Language discrimination loss (gradient reversed)
3. **Geometric Loss**: Consistency across spherical-hyperbolic spaces

## Zero-Shot Cross-Lingual Transfer

### Training Phase
- Single language (English)
- All three components optimized together
- Language adversaries learn to suppress English-specific patterns

### Testing Phase (Zero-Shot)
- Unseen languages (Chinese, Spanish, Greek, etc.)
- Language adversaries filter out language-specific information
- Geometric projections capture universal disease markers
- Model generalizes without target-language training data

## Key Innovation: Language Invariance

Traditional models fail on unseen languages because they pick up:
- Accent patterns
- Grammar structures  
- Dialect-specific vocabulary
- Prosodic patterns of the language

**ORBIT's Solution:**
- Adversarial training forces suppression of language markers
- Multimodal fusion leverages both acoustic and linguistic signals
- Bi-geometric learning finds universal representation space
- Result: Detects cognitive impairment independent of language

## Ablation Components

Each component contributes to zero-shot performance:

- **Cross-Attention Fusion**: +3.2% over concatenation
- **Language Adversaries**: +2.1% over no adversaries
- **Spherical Projection**: +1.8% over Euclidean only
- **Hyperbolic Projection**: +1.4% for hierarchical capture
- **Consensus Clustering**: +0.9% aggregation benefit

**Total improvement**: ~9.4% over baselines on zero-shot languages

## Mathematical Formulation

### Multimodal Fusion

```
s_proj = W_s @ speech
t_proj = W_t @ text

s_attended, _ = MultiheadAttention(s_proj, t_proj, t_proj)
t_attended, _ = MultiheadAttention(t_proj, s_proj, s_proj)

gate = σ(W_gate @ [s̄, t̄])
fused = gate ⊙ s̄ + (1-gate) ⊙ t̄
```

### Adversarial Loss (Gradient Reversal)

```
L_adversary = -L_discriminator(features)
              (minus sign = gradient reversal)
```

### Bi-Geometric Projection

```
x_sphere = normalize(W_s @ x)
x_hyp = tanh(W_h @ x)
x_geo = α × x_sphere + β × x_hyp
```

## Configuration

Default hyperparameters (paper values):
- Speech encoder: WavLM-Large (1024-dim, frozen)
- Text encoder: XLM-R-Large (768-dim, frozen)
- Fusion hidden: 512-dim
- Attention heads: 8
- Adversaries: 3 (multi-tap)
- Batch size: 16
- Learning rate: 1e-4 (AdamW)
- Weight decay: 1e-2
- Training epochs: 50
