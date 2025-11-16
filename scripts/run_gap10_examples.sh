#!/bin/bash
# Example training script demonstrating Gap 10 features
# This script shows different usage scenarios

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

DATASET="cifar100_ir100"
MODEL="clip_vit_b16"
BASE_TUNER="class_aware_lora"
OUTPUT_BASE="output/gap10_experiments"

# ============================================================================
# Scenario 1: Class-Aware Meta-LoRA Only (Baseline)
# ============================================================================

echo "=================================================="
echo "Scenario 1: Class-Aware Meta-LoRA Baseline"
echo "=================================================="

python main.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --tuner ${BASE_TUNER} \
  --opts \
    use_meta=True \
    use_class_aware=True \
    meta_lr=0.001 \
    num_epochs=50 \
    output_dir=${OUTPUT_BASE}/baseline

# ============================================================================
# Scenario 2: Add COCL Losses Only
# ============================================================================

echo "=================================================="
echo "Scenario 2: Class-Aware + COCL Losses"
echo "=================================================="

python main.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --tuner ${BASE_TUNER} \
  --opts \
    use_meta=True \
    use_class_aware=True \
    cocl.use_ocl=True \
    cocl.use_tail_proto=True \
    cocl.use_head_debias=True \
    cocl.lambda_ocl=0.5 \
    cocl.lambda_tail_proto=0.3 \
    cocl.lambda_head_debias=0.1 \
    num_epochs=50 \
    output_dir=${OUTPUT_BASE}/cocl_only

# ============================================================================
# Scenario 3: Add Tail Augmentation Only
# ============================================================================

echo "=================================================="
echo "Scenario 3: Class-Aware + Tail Augmentation"
echo "=================================================="

python main.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --tuner ${BASE_TUNER} \
  --opts \
    use_meta=True \
    use_class_aware=True \
    tail_augmentation.use_tail_cutmix=True \
    tail_augmentation.tail_cutmix_alpha=0.9999 \
    tail_augmentation.tail_cutmix_prob=0.5 \
    num_epochs=50 \
    output_dir=${OUTPUT_BASE}/augmentation_only

# ============================================================================
# Scenario 4: Full Gap 10 (All Features)
# ============================================================================

echo "=================================================="
echo "Scenario 4: Full Gap 10 (All Features Enabled)"
echo "=================================================="

python main.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --tuner gap10_full \
  --opts \
    num_epochs=50 \
    output_dir=${OUTPUT_BASE}/full_gap10

# Or equivalently with explicit flags:
# python main.py \
#   --dataset ${DATASET} \
#   --model ${MODEL} \
#   --tuner ${BASE_TUNER} \
#   --opts \
#     use_meta=True \
#     use_class_aware=True \
#     cocl.use_ocl=True \
#     cocl.use_tail_proto=True \
#     cocl.use_head_debias=True \
#     cocl.lambda_ocl=0.5 \
#     cocl.lambda_tail_proto=0.3 \
#     cocl.lambda_head_debias=0.1 \
#     tail_augmentation.use_tail_cutmix=True \
#     tail_augmentation.tail_cutmix_alpha=0.9999 \
#     tail_augmentation.use_ood_paste=True \
#     ood.use_ood=True \
#     ood.ood_dataset=tinyimages \
#     ood_eval.enable=True \
#     ood_eval.ood_test_datasets=[textures,svhn] \
#     num_epochs=50 \
#     output_dir=${OUTPUT_BASE}/full_gap10

# ============================================================================
# Scenario 5: OOD Evaluation on Trained Model
# ============================================================================

echo "=================================================="
echo "Scenario 5: OOD Evaluation Only"
echo "=================================================="

# Evaluate OOD detection on a previously trained model
MODEL_DIR="${OUTPUT_BASE}/full_gap10"  # Or any trained model

python main.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --tuner ${BASE_TUNER} \
  --opts \
    test_only=True \
    model_dir=${MODEL_DIR} \
    ood_eval.enable=True \
    ood_eval.ood_test_datasets=[textures,svhn,lsun] \
    ood_eval.ood_metric=energy

# ============================================================================
# Scenario 6: Generate Visualizations
# ============================================================================

echo "=================================================="
echo "Scenario 6: Generate Visualizations"
echo "=================================================="

# Generate visualizations from saved results
# (Assumes you've saved logits, labels, features during evaluation)

python scripts/analyze_results.py \
  --logits ${MODEL_DIR}/logits.npy \
  --labels ${MODEL_DIR}/labels.npy \
  --features ${MODEL_DIR}/features.npy \
  --class-counts ${MODEL_DIR}/class_counts.npy \
  --class-groups ${MODEL_DIR}/class_groups.pkl \
  --output-dir ${MODEL_DIR}/plots \
  --save-confmat \
  --save-per-class \
  --save-calibration

# Optionally include t-SNE (slow for large datasets)
# python scripts/analyze_results.py \
#   --logits ${MODEL_DIR}/logits.npy \
#   --labels ${MODEL_DIR}/labels.npy \
#   --features ${MODEL_DIR}/features.npy \
#   --ood-features ${MODEL_DIR}/ood_features.npy \
#   --output-dir ${MODEL_DIR}/plots \
#   --save-tsne

echo "=================================================="
echo "All scenarios completed!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "=================================================="
