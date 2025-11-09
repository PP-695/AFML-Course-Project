# Class-Aware Meta-Learning for Long-Tailed PEFT

## Overview

Class-Aware Meta-Learning extends the MetaLoRA framework to address the challenge of long-tailed class distributions in fine-tuning. By learning per-class LoRA hyperparameters (rank and alpha), this approach enables stronger adaptation for tail classes while preventing overfitting on head classes.

## Key Concepts

### Class Grouping

Classes are automatically divided into three groups based on sample frequency:
- **Head classes**: Abundant samples (≥100 by default)
- **Medium classes**: Moderate samples (20-100 by default)
- **Tail classes**: Few samples (<20 by default)

### Per-Class Meta-Parameters

Each class has learnable weight factors for:
- **Rank scaling**: Controls the adaptation capacity
- **Alpha scaling**: Controls the update magnitude

These are initialized differently per group:
- Head: Lower factors (0.5) - less adaptation needed
- Medium: Baseline factors (1.0)
- Tail: Higher factors (2.0) - more adaptation needed

### Meta-Optimization Objective

Instead of optimizing for overall accuracy, the meta-learner optimizes for:
- **Balanced Accuracy**: Mean of per-class accuracies
- **G-Mean**: Geometric mean of per-class accuracies
- **Worst-Case**: Minimum per-class accuracy

This ensures tail classes receive appropriate attention during meta-learning.

## Architecture

```
ClassAwareMetaLoRA (extends MetaLoRA)
├── Per-class rank weights [num_classes]
├── Per-class alpha weights [num_classes]
├── Class grouping (head/medium/tail)
└── Dynamic forward pass with class-specific scaling

ClassAwareMetaTrainer (extends MetaTrainer)
├── Class distribution analysis
├── Balanced validation metrics
├── Class-aware loss computation
├── Regularization (smoothness, divergence)
└── Visualization tools
```

## Usage

### Basic Training

```bash
python main.py \
  --dataset CIFAR100_IR100 \
  --model clip_vit_b16 \
  --tuner class_aware_lora \
  --opts \
    use_meta=True \
    use_class_aware=True \
    meta_lr=0.001 \
    num_epochs=50
```

### Configuration Options

**Meta-Learning Settings:**
```yaml
use_meta: True                    # Enable meta-learning
use_class_aware: True             # Enable class-aware features
meta_lr: 0.001                    # Learning rate for meta-parameters
meta_data_ratio: 0.2              # Fraction of data for meta-optimization
meta_update_freq: 1               # Meta-update every N epochs
meta_inner_steps: 3               # Inner loop steps per meta-update
```

**Class-Aware Settings:**
```yaml
meta_objective: balanced_accuracy # Options: balanced_accuracy, gmean, worst_case
focus_on_tail: True               # Weight tail classes higher in loss
tail_loss_weight: 2.0             # Weight multiplier for tail classes
head_threshold: 100               # Sample count threshold for head classes
tail_threshold: 20                # Sample count threshold for tail classes
```

**Initialization Factors:**
```yaml
head_rank_factor: 0.5             # Initial rank scaling for head classes
tail_rank_factor: 2.0             # Initial rank scaling for tail classes
head_alpha_factor: 0.5            # Initial alpha scaling for head classes
tail_alpha_factor: 2.0            # Initial alpha scaling for tail classes
```

**Regularization:**
```yaml
rank_divergence_penalty: 0.01     # Penalty for rank weight variance
alpha_smoothness_penalty: 0.005   # Penalty for alpha weight differences
```

### Configuration File

Create `configs/tuner/class_aware_lora.yaml`:
```yaml
use_flora: True
use_meta: True
use_class_aware: True

flora:
  arch:
    alpha: 1.0
    rank: 8

class_aware:
  num_class_groups: 3
  head_threshold: 100
  tail_threshold: 20
  head_rank_factor: 0.5
  tail_rank_factor: 2.0
  head_alpha_factor: 0.5
  tail_alpha_factor: 2.0
```

## Implementation Details

### ClassAwareMetaLoRA Module

Located in `models/class_aware_peft.py`, this module:

1. **Initialization**: 
   - Creates per-class learnable weight parameters
   - Analyzes class distribution from dataset
   - Groups classes into head/medium/tail
   - Initializes weights based on class group

2. **Forward Pass**:
   - Accepts optional class IDs for batch samples
   - Applies class-specific alpha scaling
   - Falls back to uniform scaling during inference

3. **Meta-Parameters**:
   - Returns list of learnable meta-parameters
   - Includes both base meta_alpha and per-class weights

### ClassAwareMetaTrainer

Located in `trainer_class_aware.py`, this trainer:

1. **Class Distribution Analysis**:
   - Computes imbalance metrics (IR, Gini coefficient)
   - Splits classes into groups
   - Visualizes distribution

2. **Meta-Optimization**:
   - Inner loop: Standard training on meta-train set
   - Outer loop: Evaluates on meta-val with balanced objective
   - Updates meta-parameters to improve tail performance

3. **Loss Computation**:
   - Applies higher weights to tail class samples
   - Adds regularization for parameter smoothness
   - Prevents extreme divergence

4. **Metrics Tracking**:
   - Overall accuracy
   - Balanced accuracy
   - Per-group accuracies (head/medium/tail)
   - G-Mean and worst-case accuracy

### Utility Functions

**Class Distribution Analysis** (`utils/class_imbalance_utils.py`):
- `analyze_class_distribution()`: Extract class sample counts
- `compute_imbalance_metrics()`: Calculate IR, Gini coefficient
- `split_by_frequency()` / `split_by_imbalance_ratio()`: Group classes
- `compute_balanced_accuracy()`: Calculate balanced metrics
- `visualize_class_distribution()`: Plot class distribution

**Class-Aware Losses** (`utils/class_aware_losses.py`):
- `ClassAwareLDAM`: LDAM with learnable per-class margins
- `ClassAwareBalancedSoftmax`: Balanced softmax with adjustments
- `BalancedAccuracyLoss`: Optimizes balanced accuracy
- `ClassDistributionRegularizer`: Smoothness regularization

## Expected Performance

On CIFAR100-LT (IR=100):

| Metric | Baseline | Class-Aware |
|--------|----------|-------------|
| Overall Acc | 65-70% | 66-72% |
| Head Acc | 75-80% | 74-79% |
| Medium Acc | 60-65% | 62-67% |
| Tail Acc | 35-40% | 42-48% |
| Head-Tail Gap | 40-45% | 32-37% |
| Balanced Acc | 55-60% | 62-67% |

**Key Improvements**:
- Tail accuracy: +5-10%
- Head-tail gap: -5-8%
- Balanced accuracy: +5-8%
- Overall accuracy: Maintained or slightly improved

## Hyperparameter Tuning Guide

### Meta-Learning Rate
- Start with `meta_lr=0.001`
- Increase to `0.005` if meta-parameters don't change
- Decrease to `0.0005` if training is unstable

### Meta Data Ratio
- Default: `0.2` (20% for meta-learning)
- Increase to `0.3` for more stable meta-updates
- Decrease to `0.1` to use more data for main training

### Tail Loss Weight
- Default: `2.0` (2x weight for tail classes)
- Increase to `3.0-5.0` for stronger focus on tail
- Decrease to `1.5` if head accuracy drops too much

### Initial Factors
- Default tail factors: `2.0` (2x rank and alpha)
- Increase to `3.0-4.0` for very imbalanced datasets (IR > 200)
- Decrease to `1.5` for moderately imbalanced datasets (IR < 50)

### Regularization
- `rank_divergence_penalty`: Prevents extreme variation
  - Increase if weights diverge too much
  - Decrease if adaptation is too constrained
- `alpha_smoothness_penalty`: Encourages smooth transitions
  - Increase for smoother class-to-class changes
  - Decrease to allow more flexibility

## Visualization

During training, the following visualizations are generated:

1. **Class Distribution** (`class_distribution.png`):
   - Bar plot showing samples per class
   - Color-coded by group (green=head, orange=medium, red=tail)

2. **Rank Evolution** (`rank_evolution_epoch_N.png`):
   - Per-class rank weights over time
   - Shows how adaptation capacity changes per class

3. **Alpha Evolution** (included in rank evolution):
   - Per-class alpha weights over time
   - Shows how update magnitude changes per class

## Debugging Tips

### Meta-Parameters Not Changing
- Check that `use_meta=True` and `use_class_aware=True`
- Verify meta-optimizer is being created
- Increase `meta_lr` or `meta_update_freq`
- Check gradient flow to meta-parameters

### Poor Tail Performance
- Increase `tail_loss_weight`
- Increase `tail_rank_factor` and `tail_alpha_factor`
- Decrease regularization penalties
- Try different `meta_objective` (e.g., worst_case)

### Head Performance Degradation
- Decrease `tail_loss_weight`
- Increase regularization penalties
- Adjust head/tail factors to be closer to 1.0

### Training Instability
- Decrease `meta_lr`
- Increase regularization penalties
- Use smaller `meta_inner_steps`
- Enable gradient clipping

## Integration with Existing Code

The class-aware framework is designed to be minimally invasive:

1. **Model Building** (`models/peft_vit.py`):
   - Automatically uses ClassAwareMetaLoRA when `use_class_aware=True`
   - Falls back to standard FLoRA otherwise

2. **Trainer Selection** (`main.py`):
   - Automatically selects ClassAwareMetaTrainer when `use_class_aware=True`
   - Falls back to MetaTrainer/Trainer otherwise

3. **Configuration** (`utils/config_omega.py`):
   - All class-aware options have sensible defaults
   - Can be overridden via command line or config files

## References

- MetaLoRA: Meta-Learning Hyperparameters for PEFT (base framework)
- LDAM: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
- Balanced Softmax: Balanced Meta-Softmax for Long-Tailed Visual Recognition
- FLoRA: Fine-grained LoRA for parameter-efficient fine-tuning

## Future Extensions

Potential improvements:
- Dynamic class grouping during training
- Attention-based class similarity for smoother transitions
- Multi-level hierarchy beyond head/medium/tail
- Integration with other PEFT methods (Adapter, VPT)
- Transfer learning across datasets with different imbalance ratios
