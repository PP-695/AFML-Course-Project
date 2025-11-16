#!/usr/bin/env python3
"""
Smoke tests for Gap 10 components.

Tests basic functionality without requiring full training runs.
"""

import sys
import torch
import numpy as np


def test_class_aware_losses():
    """Test COCL-style loss components."""
    print("Testing class-aware losses...")
    
    from utils.class_aware_losses import (
        OutlierClassLoss,
        TailPrototypeLoss,
        DebiasedHeadLoss,
        calibrate_logits
    )
    
    # Test OutlierClassLoss
    print("  - OutlierClassLoss")
    ocl = OutlierClassLoss(num_id_classes=10, weight=1.0)
    logits = torch.randn(4, 11)  # 10 ID classes + 1 outlier
    labels = torch.tensor([0, 1, 10, 10])  # Last two are OOD
    loss = ocl(logits, labels)
    assert loss.item() >= 0, "OCL loss should be non-negative"
    
    # Test TailPrototypeLoss
    print("  - TailPrototypeLoss")
    tpl = TailPrototypeLoss(temperature=0.07, margin=0.1)
    tail_features = torch.randn(5, 128)
    ood_features = torch.randn(5, 128)
    tail_labels = torch.tensor([0, 0, 1, 1, 2])
    loss = tpl(tail_features, ood_features, tail_labels)
    assert loss.item() >= 0, "Tail prototype loss should be non-negative"
    
    # Test DebiasedHeadLoss
    print("  - DebiasedHeadLoss")
    cls_num_list = [500, 400, 300, 50, 30, 20, 10, 5]
    dhl = DebiasedHeadLoss(cls_num_list, head_threshold=100, penalty_weight=0.1)
    logits = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 5, 7])
    loss = dhl(logits, labels)
    assert loss.item() >= 0, "Debiased head loss should be non-negative"
    
    # Test calibrate_logits
    print("  - calibrate_logits")
    logits = torch.randn(4, 5)
    class_priors = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    calibrated = calibrate_logits(logits, class_priors, tau=1.0)
    assert calibrated.shape == logits.shape, "Shape should be preserved"
    
    print("  ✓ All loss components work correctly\n")


def test_tail_augmentation():
    """Test tail augmentation components."""
    print("Testing tail augmentation...")
    
    from datasets.tail_augmentation import (
        rand_bbox,
        augment_tail_with_cutmix,
        TailAugmentationMixer,
        get_tail_mask
    )
    
    # Test rand_bbox
    print("  - rand_bbox")
    size = (4, 3, 224, 224)
    bbox = rand_bbox(size, lam=0.5)
    assert len(bbox) == 4, "Should return 4 coordinates"
    assert all(0 <= x <= 224 for x in bbox), "Coordinates should be in bounds"
    
    # Test augment_tail_with_cutmix
    print("  - augment_tail_with_cutmix")
    tail_images = torch.randn(4, 3, 224, 224)
    tail_labels = torch.tensor([5, 6, 7, 8])
    bg_images = torch.randn(10, 3, 224, 224)
    mixed, labels, lam = augment_tail_with_cutmix(
        tail_images, tail_labels, bg_images, alpha=0.9999
    )
    assert mixed.shape == tail_images.shape, "Shape should be preserved"
    assert torch.equal(labels, tail_labels), "Labels should remain tail labels"
    
    # Test TailAugmentationMixer
    print("  - TailAugmentationMixer")
    mixer = TailAugmentationMixer(alpha=0.9999, prob=1.0, use_ood=False)
    head_buffer = (torch.randn(10, 3, 224, 224), torch.tensor([0, 1, 2] * 3 + [0]))
    mixer.set_head_buffer(*head_buffer)
    
    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 1, 5, 6, 7, 8, 9, 2])
    is_tail = torch.tensor([False, False, True, True, True, True, True, False])
    mixed_images, mixed_labels = mixer(images, labels, is_tail)
    assert mixed_images.shape == images.shape, "Shape should be preserved"
    
    # Test get_tail_mask
    print("  - get_tail_mask")
    labels = torch.tensor([0, 1, 5, 6, 7])
    tail_indices = [5, 6, 7, 8, 9]
    mask = get_tail_mask(labels, tail_indices)
    expected = torch.tensor([False, False, True, True, True])
    assert torch.equal(mask, expected), "Tail mask should be correct"
    
    print("  ✓ All augmentation components work correctly\n")


def test_ood_eval():
    """Test OOD evaluation components."""
    print("Testing OOD evaluation...")
    
    from utils.ood_eval import (
        compute_ood_scores_msp,
        compute_ood_scores_energy,
        compute_metrics,
        compute_fpr_at_tpr
    )
    
    # Test MSP scores
    print("  - compute_ood_scores_msp")
    logits = torch.randn(10, 5)
    scores = compute_ood_scores_msp(logits)
    assert len(scores) == 10, "Should have scores for all samples"
    assert all(0 <= s <= 1 for s in scores), "MSP scores should be in [0, 1]"
    
    # Test Energy scores
    print("  - compute_ood_scores_energy")
    scores = compute_ood_scores_energy(logits, temperature=1.0)
    assert len(scores) == 10, "Should have scores for all samples"
    
    # Test metrics
    print("  - compute_metrics")
    id_scores = np.random.rand(100)
    ood_scores = np.random.rand(100) * 0.5  # Lower scores for OOD
    metrics = compute_metrics(id_scores, ood_scores)
    assert 'auroc' in metrics, "Should have AUROC"
    assert 'aupr_in' in metrics, "Should have AUPR-IN"
    assert 'fpr95' in metrics, "Should have FPR95"
    assert 0 <= metrics['auroc'] <= 1, "AUROC should be in [0, 1]"
    
    # Test FPR@TPR
    print("  - compute_fpr_at_tpr")
    y_true = np.array([1] * 50 + [0] * 50)
    y_score = np.concatenate([np.random.rand(50) + 0.5, np.random.rand(50)])
    fpr95 = compute_fpr_at_tpr(y_true, y_score, tpr_threshold=0.95)
    assert 0 <= fpr95 <= 1, "FPR should be in [0, 1]"
    
    print("  ✓ All OOD evaluation components work correctly\n")


def test_visualization():
    """Test visualization components."""
    print("Testing visualization...")
    
    from scripts.analyze_results import (
        compute_per_class_accuracy,
    )
    
    # Test per-class accuracy computation
    print("  - compute_per_class_accuracy")
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 3, 3])
    accs = compute_per_class_accuracy(y_true, y_pred, num_classes=4)
    assert len(accs) == 4, "Should have accuracy for each class"
    assert accs[0] == 0.5, "Class 0 accuracy should be 0.5"
    assert accs[1] == 1.0, "Class 1 accuracy should be 1.0"
    assert accs[2] == 0.5, "Class 2 accuracy should be 0.5"
    assert accs[3] == 1.0, "Class 3 accuracy should be 1.0"
    
    print("  ✓ Visualization components work correctly\n")


def main():
    """Run all smoke tests."""
    print("="*60)
    print("Gap 10 Components Smoke Tests")
    print("="*60 + "\n")
    
    try:
        test_class_aware_losses()
        test_tail_augmentation()
        test_ood_eval()
        test_visualization()
        
        print("="*60)
        print("✓ All smoke tests passed!")
        print("="*60)
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
