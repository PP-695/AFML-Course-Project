"""
Utility functions for analyzing and handling class imbalance in long-tailed datasets.

This module provides tools for:
- Analyzing class distributions
- Computing imbalance metrics
- Splitting classes into head/medium/tail groups
- Visualizing class distributions
- Computing balanced accuracy metrics
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


def analyze_class_distribution(dataset) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Analyze the class distribution of a dataset.
    
    Args:
        dataset: Dataset object with cls_num_list attribute
        
    Returns:
        cls_num_list: Array of sample counts per class
        metrics: Dictionary of imbalance metrics
    """
    cls_num_list = np.array(dataset.cls_num_list)
    metrics = compute_imbalance_metrics(cls_num_list)
    
    return cls_num_list, metrics


def compute_imbalance_metrics(cls_num_list: np.ndarray) -> Dict[str, float]:
    """
    Compute various imbalance metrics for a class distribution.
    
    Args:
        cls_num_list: Array of sample counts per class
        
    Returns:
        Dictionary containing:
        - imbalance_ratio: max_samples / min_samples
        - gini_coefficient: Gini coefficient of the distribution
        - mean_samples: Mean number of samples per class
        - std_samples: Standard deviation of samples per class
    """
    cls_num_list = np.array(cls_num_list)
    
    # Imbalance Ratio (IR)
    imbalance_ratio = cls_num_list.max() / cls_num_list.min()
    
    # Gini coefficient
    sorted_counts = np.sort(cls_num_list)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts))) / (n * cumsum[-1]) - (n + 1) / n
    
    return {
        'imbalance_ratio': float(imbalance_ratio),
        'gini_coefficient': float(gini),
        'mean_samples': float(cls_num_list.mean()),
        'std_samples': float(cls_num_list.std()),
        'min_samples': int(cls_num_list.min()),
        'max_samples': int(cls_num_list.max()),
        'num_classes': len(cls_num_list)
    }


def split_by_frequency(
    cls_num_list: np.ndarray,
    head_ratio: float = 0.3,
    tail_ratio: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split classes into head/medium/tail groups based on sample frequency.
    
    Args:
        cls_num_list: Array of sample counts per class
        head_ratio: Fraction of classes in the head group (by sample count)
        tail_ratio: Fraction of classes in the tail group (by sample count)
        
    Returns:
        head_indices: Indices of head classes
        medium_indices: Indices of medium classes
        tail_indices: Indices of tail classes
    """
    cls_num_list = np.array(cls_num_list)
    num_classes = len(cls_num_list)
    
    # Sort classes by sample count (descending)
    sorted_indices = np.argsort(cls_num_list)[::-1]
    
    # Compute cumulative sample counts
    total_samples = cls_num_list.sum()
    cumsum = np.cumsum(cls_num_list[sorted_indices])
    
    # Find split points based on cumulative sample ratio
    head_threshold = total_samples * head_ratio
    tail_threshold = total_samples * (1 - tail_ratio)
    
    head_split = np.searchsorted(cumsum, head_threshold) + 1
    tail_split = np.searchsorted(cumsum, tail_threshold) + 1
    
    head_indices = sorted_indices[:head_split]
    medium_indices = sorted_indices[head_split:tail_split]
    tail_indices = sorted_indices[tail_split:]
    
    return head_indices, medium_indices, tail_indices


def split_by_imbalance_ratio(
    cls_num_list: np.ndarray,
    head_threshold: int = 100,
    tail_threshold: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split classes into head/medium/tail groups based on absolute sample counts.
    
    Args:
        cls_num_list: Array of sample counts per class
        head_threshold: Minimum samples for head classes (>= threshold)
        tail_threshold: Maximum samples for tail classes (< threshold)
        
    Returns:
        head_indices: Indices of head classes (>= head_threshold)
        medium_indices: Indices of medium classes (>= tail_threshold and < head_threshold)
        tail_indices: Indices of tail classes (< tail_threshold)
    """
    cls_num_list = np.array(cls_num_list)
    
    head_indices = np.where(cls_num_list >= head_threshold)[0]
    tail_indices = np.where(cls_num_list < tail_threshold)[0]
    medium_indices = np.where((cls_num_list >= tail_threshold) & (cls_num_list < head_threshold))[0]
    
    return head_indices, medium_indices, tail_indices


def compute_balanced_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute balanced accuracy metrics including per-class accuracies.
    
    Args:
        preds: Predicted class labels
        labels: Ground truth labels
        num_classes: Total number of classes
        
    Returns:
        Dictionary containing:
        - overall_acc: Overall accuracy
        - balanced_acc: Balanced accuracy (mean of per-class accuracies)
        - geometric_mean: Geometric mean of per-class accuracies
        - worst_case_acc: Minimum per-class accuracy
        - per_class_acc: Array of per-class accuracies
    """
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    # Compute per-class accuracy
    per_class_acc = np.zeros(num_classes)
    per_class_count = np.zeros(num_classes)
    
    for i in range(num_classes):
        mask = labels == i
        per_class_count[i] = mask.sum()
        if per_class_count[i] > 0:
            per_class_acc[i] = (preds[mask] == labels[mask]).sum() / per_class_count[i]
    
    # Overall accuracy
    overall_acc = (preds == labels).mean()
    
    # Balanced accuracy (mean of per-class accuracies)
    # Only consider classes that have samples
    valid_classes = per_class_count > 0
    balanced_acc = per_class_acc[valid_classes].mean()
    
    # Geometric mean (G-Mean)
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    geometric_mean = np.exp(np.log(per_class_acc[valid_classes] + eps).mean())
    
    # Worst-case accuracy
    worst_case_acc = per_class_acc[valid_classes].min() if valid_classes.any() else 0.0
    
    return {
        'overall_acc': float(overall_acc),
        'balanced_acc': float(balanced_acc),
        'geometric_mean': float(geometric_mean),
        'worst_case_acc': float(worst_case_acc),
        'per_class_acc': per_class_acc
    }


def compute_group_accuracy(
    preds: torch.Tensor,
    labels: torch.Tensor,
    head_indices: np.ndarray,
    medium_indices: np.ndarray,
    tail_indices: np.ndarray
) -> Dict[str, float]:
    """
    Compute accuracy for each class group (head/medium/tail).
    
    Args:
        preds: Predicted class labels
        labels: Ground truth labels
        head_indices: Indices of head classes
        medium_indices: Indices of medium classes
        tail_indices: Indices of tail classes
        
    Returns:
        Dictionary containing accuracies for each group
    """
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    def group_acc(indices):
        if len(indices) == 0:
            return 0.0
        mask = np.isin(labels, indices)
        if mask.sum() == 0:
            return 0.0
        return (preds[mask] == labels[mask]).mean()
    
    return {
        'head_acc': float(group_acc(head_indices)),
        'medium_acc': float(group_acc(medium_indices)),
        'tail_acc': float(group_acc(tail_indices))
    }


def visualize_class_distribution(
    cls_num_list: np.ndarray,
    head_indices: Optional[np.ndarray] = None,
    medium_indices: Optional[np.ndarray] = None,
    tail_indices: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize the class distribution with optional head/medium/tail coloring.
    
    Args:
        cls_num_list: Array of sample counts per class
        head_indices: Indices of head classes
        medium_indices: Indices of medium classes
        tail_indices: Indices of tail classes
        save_path: Path to save the figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    num_classes = len(cls_num_list)
    colors = ['gray'] * num_classes
    
    if head_indices is not None:
        for idx in head_indices:
            colors[idx] = 'green'
    if medium_indices is not None:
        for idx in medium_indices:
            colors[idx] = 'orange'
    if tail_indices is not None:
        for idx in tail_indices:
            colors[idx] = 'red'
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_classes), cls_num_list, color=colors, alpha=0.7)
    plt.xlabel('Class Index')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    if head_indices is not None and len(head_indices) > 0:
        legend_elements.append(Patch(facecolor='green', alpha=0.7, label='Head'))
    if medium_indices is not None and len(medium_indices) > 0:
        legend_elements.append(Patch(facecolor='orange', alpha=0.7, label='Medium'))
    if tail_indices is not None and len(tail_indices) > 0:
        legend_elements.append(Patch(facecolor='red', alpha=0.7, label='Tail'))
    
    if legend_elements:
        plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved class distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
