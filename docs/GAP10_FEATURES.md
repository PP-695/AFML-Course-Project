# Gap 10: Advanced Long-Tailed OOD Learning Features

## Overview

This document describes the Gap 10 extensions to the Class-Aware Meta-LoRA framework, which add state-of-the-art components for long-tailed learning with out-of-distribution (OOD) detection.

## Features

### 1. COCL-Style Loss Components

Three advanced loss components inspired by recent work on long-tailed learning and OOD detection:

#### Outlier Class Learning (OCL)
- **Purpose**: Explicitly model OOD samples with an auxiliary outlier class
- **Mechanism**: Extends the classifier with one additional class (index `num_classes`) reserved for outliers
- **Usage**: Requires auxiliary OOD data during training
- **Config**: `cocl.use_ocl=True`, `cocl.lambda_ocl=0.5`

#### Tail Prototype Learning
- **Purpose**: Push tail class representations away from OOD distribution
- **Mechanism**: Contrastive learning that maximizes distance between tail prototypes and OOD features
- **Usage**: Works with features extracted before the classifier
- **Config**: `cocl.use_tail_proto=True`, `cocl.lambda_tail_proto=0.3`

#### Debiased Head Loss
- **Purpose**: Reduce over-confidence in head classes for better calibration
- **Mechanism**: Applies entropy-based penalty to discourage extreme confidence on head classes
- **Usage**: Applied automatically to all samples
- **Config**: `cocl.use_head_debias=True`, `cocl.lambda_head_debias=0.1`

#### Logit Calibration
- **Purpose**: Improve prediction confidence estimates at test time
- **Mechanism**: Adjusts logits based on class priors: `logits - tau * log(prior)`
- **Usage**: Applied at inference time only
- **Config**: `cocl.use_logit_calibration=True`, `cocl.tau_calibrate=1.0`

### 2. EAT-Style Tail Augmentation

CutMix-based augmentation that focuses on improving tail class representations:

- **Mechanism**: Pastes tail class patches into head or OOD images
- **Beta Parameter**: High alpha (e.g., 0.9999) preserves most of the tail image
- **Mixing Options**: 
  - Tail + Head: Mix with head class images (improves tail-head separation)
  - Tail + OOD: Mix with OOD images (improves tail-OOD separation)
- **Config**: `tail_augmentation.use_tail_cutmix=True`

### 3. OOD Detection Evaluation

Comprehensive OOD detection metrics and support for multiple benchmarks:

#### Supported OOD Scores
- **MSP (Maximum Softmax Probability)**: Fast, simple baseline
- **Energy**: Often better than MSP, same computational cost
- **ODIN**: Best performance but requires gradient computation

#### Metrics
- **AUROC**: Area Under ROC Curve (higher is better)
- **AUPR-IN**: Area Under PR Curve with ID as positive
- **AUPR-OUT**: Area Under PR Curve with OOD as positive
- **FPR@95**: False Positive Rate at 95% True Positive Rate (lower is better)

#### Supported OOD Datasets
- **TinyImages**: 300K random images (common benchmark)
- **Places365**: High-resolution scene images
- **LSUN**: Large-scale scene understanding
- **Textures**: Describable Textures Dataset (DTD)
- **SVHN**: Street View House Numbers
- **Gaussian**: Synthetic Gaussian noise
- **Uniform**: Synthetic uniform noise

### 4. Advanced Visualizations

Comprehensive analysis tools for understanding model behavior:

#### Confusion Matrix Heatmaps
- Visualize per-class prediction patterns
- Identify common confusion pairs
- Normalized or absolute counts

#### Per-Class Accuracy Plots
- Scatter plot: accuracy vs. sample count
- Color-coded by class group (head/medium/tail)
- Log-scale x-axis for wide range of counts

#### t-SNE Embeddings
- 2D visualization of feature space
- Separate colors for head/medium/tail/OOD
- Helps verify feature separation

#### Calibration Curves
- Reliability diagrams per class group
- Compare predicted confidence vs. actual accuracy
- Identifies over/under-confidence issues

## Installation

No additional dependencies required beyond the base MetaLoRA framework:

```bash
# All Gap 10 features use existing dependencies
pip install torch torchvision matplotlib seaborn scikit-learn
```

## Quick Start

### Basic Usage (COCL Losses Only)

```bash
python main.py \
  --dataset cifar100_ir100 \
  --model clip_vit_b16 \
  --tuner class_aware_lora \
  --opts \
    use_meta=True \
    use_class_aware=True \
    cocl.use_ocl=True \
    cocl.use_tail_proto=True \
    cocl.use_head_debias=True
```

### Full Gap 10 Configuration

```bash
python main.py \
  --dataset cifar100_ir100 \
  --model clip_vit_b16 \
  --tuner gap10_full  # Uses configs/tuner/gap10_full.yaml
```

Or with explicit options:

```bash
python main.py \
  --dataset cifar100_ir100 \
  --model clip_vit_b16 \
  --tuner class_aware_lora \
  --opts \
    use_meta=True \
    use_class_aware=True \
    cocl.use_ocl=True \
    cocl.use_tail_proto=True \
    cocl.use_head_debias=True \
    tail_augmentation.use_tail_cutmix=True \
    ood.use_ood=True \
    ood.ood_dataset=tinyimages \
    ood_eval.enable=True \
    ood_eval.ood_test_datasets=[textures,svhn]
```

### OOD Evaluation Only

```bash
python main.py \
  --dataset cifar100_ir100 \
  --model clip_vit_b16 \
  --tuner class_aware_lora \
  --opts \
    test_only=True \
    model_dir=output/your_experiment \
    ood_eval.enable=True \
    ood_eval.ood_test_datasets=[textures,svhn,lsun] \
    ood_eval.ood_metric=energy
```

### Generate Visualizations

```bash
python scripts/analyze_results.py \
  --logits output/experiment/logits.npy \
  --labels output/experiment/labels.npy \
  --features output/experiment/features.npy \
  --class-counts output/experiment/class_counts.npy \
  --output-dir output/plots \
  --save-confmat \
  --save-per-class \
  --save-calibration
```

## Configuration Reference

### COCL Configuration

```yaml
cocl:
  use_ocl: True
  use_tail_proto: True
  use_head_debias: True
  lambda_ocl: 0.5              # Loss weight for OCL
  lambda_tail_proto: 0.3       # Loss weight for tail prototype
  lambda_head_debias: 0.1      # Loss weight for head debiasing
  tail_proto_temperature: 0.07 # Temperature for contrastive learning
  tail_proto_margin: 0.1       # Margin for tail-OOD separation
  use_logit_calibration: True  # Enable at test time
  tau_calibrate: 1.0           # Calibration strength
```

### Tail Augmentation Configuration

```yaml
tail_augmentation:
  use_tail_cutmix: True
  tail_cutmix_alpha: 0.9999    # Beta parameter (high = more tail)
  tail_cutmix_prob: 0.5        # Probability of applying
  use_ood_paste: True          # Use OOD as background
```

### OOD Configuration

```yaml
ood:
  use_ood: True
  ood_dataset: "tinyimages"    # Dataset name or path
  ood_data_path: "./data"      # Root data directory
  ood_batch_size: 32
  ood_num_samples: 10000       # 0 = use all
```

### OOD Evaluation Configuration

```yaml
ood_eval:
  enable: True
  ood_test_datasets: ["textures", "svhn", "lsun"]
  ood_metric: "msp"            # or "energy", "odin"
  compute_auroc: True
  compute_aupr: True
  compute_fpr95: True
```

## Expected Performance

### CIFAR100-LT (IR=100)

**Class-Aware Meta-LoRA (Base):**
- Tail accuracy: +5-10% over standard methods
- Balanced accuracy: +5-8%
- Head-tail gap: -5-8%

**Gap 10 Extensions (Full):**
- OOD Detection AUROC: 85-90% (vs. Textures, SVHN)
- Tail class ECE: Improved by 3-5%
- Head over-confidence: Reduced by 5-10%
- Additional balanced accuracy: +2-3% over base

## Troubleshooting

### OOD Data Not Found

If you see "OOD dataset not found" errors:

1. Check that `ood_data_path` points to the correct directory
2. Ensure OOD dataset is downloaded and extracted
3. For TinyImages, LSUN, etc., follow dataset-specific setup instructions
4. Use synthetic OOD (`gaussian` or `uniform`) for testing without downloading

### Memory Issues with t-SNE

t-SNE is computationally expensive:

1. Disable t-SNE for large datasets: `visualization.save_tsne=False`
2. Reduce number of samples used for t-SNE
3. Use UMAP instead (requires `pip install umap-learn`)

### Low OOD Detection Performance

Try these improvements:

1. Use Energy scores instead of MSP: `ood_eval.ood_metric=energy`
2. Increase tail prototype loss weight: `cocl.lambda_tail_proto=0.5`
3. Enable OOD-paste augmentation: `tail_augmentation.use_ood_paste=True`
4. Adjust temperature: `cocl.tail_proto_temperature=0.05`

## File Structure

```
metalora/
├── configs/tuner/
│   ├── class_aware_lora.yaml      # Base config
│   └── gap10_full.yaml            # Full Gap 10 config
├── datasets/
│   ├── tail_augmentation.py       # CutMix for tail classes
│   └── ood_sampler.py             # OOD data loading
├── utils/
│   ├── class_aware_losses.py      # COCL losses + calibration
│   └── ood_eval.py                # OOD detection metrics
├── scripts/
│   ├── analyze_results.py         # Visualization tools
│   ├── test_gap10.py              # Smoke tests
│   └── run_gap10_examples.sh      # Usage examples
├── trainer_class_aware.py         # Gap 10 integration
└── docs/
    └── GAP10_FEATURES.md          # This file
```

## Citation

If you use Gap 10 features in your research, please cite:

```bibtex
@InProceedings{Tian_2025_CVPR,
    author    = {Tian, Zichen and Liu, Yaoyao and Sun, Qianru},
    title     = {Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {23037-23047}
}
```

## References

- **COCL**: "Calibrated OOD Class Learning for Long-Tailed Recognition"
- **EAT**: "Effective Tail Augmentation for Long-Tailed Recognition"
- **ODIN**: "Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks"
- **Energy**: "Energy-based Out-of-distribution Detection"
