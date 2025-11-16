# Gap 10 Quick Reference

## What is Gap 10?

Gap 10 extends the Class-Aware Meta-LoRA framework with state-of-the-art components for:
- Better tail class learning via augmentation
- OOD detection capabilities
- Improved calibration and confidence estimates
- Comprehensive analysis tools

## Quick Start Commands

### 1. Basic Training (Add COCL Losses)

```bash
python main.py --dataset cifar100_ir100 --model clip_vit_b16 --tuner class_aware_lora \
  --opts cocl.use_ocl=True cocl.use_tail_proto=True cocl.use_head_debias=True
```

### 2. Full Gap 10 Training

```bash
python main.py --dataset cifar100_ir100 --model clip_vit_b16 --tuner gap10_full
```

### 3. Evaluate OOD Detection

```bash
python main.py --dataset cifar100_ir100 --model clip_vit_b16 --tuner class_aware_lora \
  --opts test_only=True model_dir=output/your_exp ood_eval.enable=True \
         ood_eval.ood_test_datasets=[textures,svhn]
```

### 4. Generate Visualizations

```bash
python scripts/analyze_results.py --logits output/exp/logits.npy \
  --labels output/exp/labels.npy --save-confmat --save-per-class --save-calibration
```

## Key Configuration Parameters

### COCL Losses
```yaml
cocl:
  use_ocl: True                    # Enable outlier class learning
  use_tail_proto: True             # Enable tail prototype learning
  use_head_debias: True            # Enable head debiasing
  lambda_ocl: 0.5                  # OCL loss weight
  lambda_tail_proto: 0.3           # Tail proto loss weight
  lambda_head_debias: 0.1          # Head debias loss weight
```

### Tail Augmentation
```yaml
tail_augmentation:
  use_tail_cutmix: True            # Enable CutMix for tail
  tail_cutmix_alpha: 0.9999        # Beta parameter (high = preserve tail)
  use_ood_paste: True              # Mix with OOD backgrounds
```

### OOD Data
```yaml
ood:
  use_ood: True                    # Use OOD data in training
  ood_dataset: "tinyimages"        # Dataset name or path
  ood_batch_size: 32               # Batch size for OOD
```

### OOD Evaluation
```yaml
ood_eval:
  enable: True                     # Enable OOD metrics
  ood_test_datasets: ["textures", "svhn", "lsun"]
  ood_metric: "msp"                # "msp", "energy", or "odin"
```

## File Reference

### Core Implementations
- `utils/class_aware_losses.py` - COCL losses (OCL, TailProto, DebiasHead, calibration)
- `datasets/tail_augmentation.py` - EAT-style CutMix augmentation
- `datasets/ood_sampler.py` - OOD data loading (TinyImages, Places365, etc.)
- `utils/ood_eval.py` - OOD detection metrics (AUROC, AUPR, FPR95)
- `scripts/analyze_results.py` - Visualization tools
- `trainer_class_aware.py` - Gap 10 integration

### Configuration
- `configs/tuner/gap10_full.yaml` - Full Gap 10 config (recommended)
- `configs/tuner/class_aware_lora.yaml` - Extended base config

### Documentation
- `docs/GAP10_FEATURES.md` - Comprehensive feature guide
- `README.md` - Gap 10 section with examples
- `IMPLEMENTATION_SUMMARY.md` - Technical details

### Examples & Tests
- `scripts/run_gap10_examples.sh` - Usage examples
- `scripts/test_gap10.py` - Smoke tests

## Common Use Cases

### 1. Improve Tail Class Accuracy
Enable tail augmentation:
```bash
--opts tail_augmentation.use_tail_cutmix=True tail_augmentation.use_ood_paste=True
```

### 2. Better Calibration
Enable head debiasing and logit calibration:
```bash
--opts cocl.use_head_debias=True cocl.use_logit_calibration=True
```

### 3. OOD Detection
Enable OCL and tail prototype learning:
```bash
--opts cocl.use_ocl=True cocl.use_tail_proto=True ood.use_ood=True ood.ood_dataset=tinyimages
```

### 4. Full Analysis
Enable all visualizations:
```bash
--opts visualization.enable=True visualization.save_confmat=True \
       visualization.save_per_class=True visualization.save_calibration=True
```

## Troubleshooting

### Issue: OOD dataset not found
**Solution**: Check `ood.ood_data_path` points to correct directory, or use synthetic OOD:
```bash
--opts ood.ood_dataset=gaussian
```

### Issue: Out of memory
**Solution**: Reduce OOD batch size or disable t-SNE:
```bash
--opts ood.ood_batch_size=16 visualization.save_tsne=False
```

### Issue: Low OOD detection performance
**Solution**: Try energy scores instead of MSP:
```bash
--opts ood_eval.ood_metric=energy
```

## Expected Performance Gains

On CIFAR100-LT (IR=100) vs. baseline Class-Aware Meta-LoRA:
- **Tail accuracy**: +2-3%
- **OOD Detection AUROC**: 85-90%
- **Calibration (ECE)**: -3-5% (lower is better)
- **Balanced accuracy**: +2-3%

## Support

For detailed documentation, see:
- `docs/GAP10_FEATURES.md` - Complete feature guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- Example scripts in `scripts/run_gap10_examples.sh`
