"""
Class-Aware Meta-Learning for Parameter-Efficient Fine-Tuning.

This module implements ClassAwareMetaLoRA, which extends MetaLoRA to support
per-class meta-parameters (rank and alpha) for improved long-tailed learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from models.peft_modules import MetaLoRA


class ClassAwareMetaLoRA(MetaLoRA):
    """
    LoRA with class-aware meta-parameters for long-tailed learning.
    
    This module extends MetaLoRA by introducing per-class learnable parameters
    that control the rank and alpha scaling for different class groups.
    Classes are grouped into head/medium/tail based on their sample frequency.
    
    Args:
        in_dim: Input dimension
        bottle_dim: Base LoRA rank
        alpha: Base alpha value for scaling
        dtype: Data type for parameters
        use_meta: Enable meta-learning
        num_classes: Total number of classes
        class_grouping: Strategy for grouping ('frequency' or 'threshold')
        num_class_groups: Number of class groups (typically 3: head/medium/tail)
        cls_num_list: Optional array of sample counts per class for initialization
    """
    
    def __init__(
        self,
        in_dim: int,
        bottle_dim: int,
        alpha: float = 1.0,
        dtype=None,
        use_meta: bool = False,
        num_classes: int = 100,
        class_grouping: str = 'frequency',
        num_class_groups: int = 3,
        cls_num_list: Optional[np.ndarray] = None
    ):
        # Initialize parent MetaLoRA
        super().__init__(in_dim, bottle_dim, alpha, dtype, use_meta)
        
        self.num_classes = num_classes
        self.class_grouping = class_grouping
        self.num_class_groups = num_class_groups
        
        # Per-class meta-parameters
        if use_meta:
            # Class-specific rank adjustments (multiplicative factors)
            # Initialize to 1.0 (no adjustment)
            self.class_rank_weights = nn.Parameter(torch.ones(num_classes, dtype=dtype))
            
            # Class-specific alpha adjustments (multiplicative factors)
            # Initialize to 1.0 (no adjustment)
            self.class_alpha_weights = nn.Parameter(torch.ones(num_classes, dtype=dtype))
        else:
            self.class_rank_weights = None
            self.class_alpha_weights = None
        
        # Class grouping information (set via initialize_from_imbalance)
        self.register_buffer('class_groups', torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer('head_indices', torch.tensor([], dtype=torch.long))
        self.register_buffer('medium_indices', torch.tensor([], dtype=torch.long))
        self.register_buffer('tail_indices', torch.tensor([], dtype=torch.long))
        
        # Initialize from class distribution if provided
        if cls_num_list is not None:
            self.initialize_from_imbalance(cls_num_list)
    
    def initialize_from_imbalance(
        self,
        cls_num_list: np.ndarray,
        head_threshold: int = 100,
        tail_threshold: int = 20,
        head_rank_factor: float = 0.5,
        tail_rank_factor: float = 2.0,
        head_alpha_factor: float = 0.5,
        tail_alpha_factor: float = 2.0
    ):
        """
        Initialize class-specific parameters based on class imbalance.
        
        Tail classes (fewer samples) get higher rank and alpha factors to
        enable stronger adaptation, while head classes get lower factors.
        
        Args:
            cls_num_list: Array of sample counts per class
            head_threshold: Minimum samples for head classes
            tail_threshold: Maximum samples for tail classes  
            head_rank_factor: Rank multiplier for head classes
            tail_rank_factor: Rank multiplier for tail classes
            head_alpha_factor: Alpha multiplier for head classes
            tail_alpha_factor: Alpha multiplier for tail classes
        """
        cls_num_list = np.array(cls_num_list)
        
        # Split classes into groups
        head_mask = cls_num_list >= head_threshold
        tail_mask = cls_num_list < tail_threshold
        medium_mask = ~(head_mask | tail_mask)
        
        # Store group indices
        self.head_indices = torch.tensor(np.where(head_mask)[0], dtype=torch.long)
        self.medium_indices = torch.tensor(np.where(medium_mask)[0], dtype=torch.long)
        self.tail_indices = torch.tensor(np.where(tail_mask)[0], dtype=torch.long)
        
        # Create class group labels (0=head, 1=medium, 2=tail)
        class_groups = torch.zeros(self.num_classes, dtype=torch.long)
        class_groups[self.head_indices] = 0
        class_groups[self.medium_indices] = 1
        class_groups[self.tail_indices] = 2
        self.class_groups = class_groups
        
        # Initialize class-specific weights if using meta-learning
        if self.class_rank_weights is not None:
            with torch.no_grad():
                # Initialize rank weights
                self.class_rank_weights[self.head_indices] = head_rank_factor
                self.class_rank_weights[self.medium_indices] = 1.0
                self.class_rank_weights[self.tail_indices] = tail_rank_factor
                
                # Initialize alpha weights
                self.class_alpha_weights[self.head_indices] = head_alpha_factor
                self.class_alpha_weights[self.medium_indices] = 1.0
                self.class_alpha_weights[self.tail_indices] = tail_alpha_factor
        
        print(f"Initialized ClassAwareMetaLoRA:")
        print(f"  Head classes ({len(self.head_indices)}): rank×{head_rank_factor}, alpha×{head_alpha_factor}")
        print(f"  Medium classes ({len(self.medium_indices)}): rank×1.0, alpha×1.0")
        print(f"  Tail classes ({len(self.tail_indices)}): rank×{tail_rank_factor}, alpha×{tail_alpha_factor}")
    
    def forward(self, x, class_ids: Optional[torch.Tensor] = None):
        """
        Forward pass with class-aware scaling.
        
        Args:
            x: Input tensor
            class_ids: Optional class IDs for current batch (batch_size,)
                      If None, uses uniform scaling (inference mode)
        
        Returns:
            LoRA output with class-specific scaling
        """
        # Standard LoRA computation
        lora_output = x @ self.lora_A @ self.lora_B
        
        # Determine scaling factor
        if self.use_meta and self.meta_alpha is not None:
            base_scaling = self.meta_alpha / self.lora_A.size(1)
        else:
            base_scaling = self.alpha / self.lora_A.size(1)
        
        # Apply class-specific scaling if available and class_ids provided
        if class_ids is not None and self.class_alpha_weights is not None:
            # Get per-sample alpha weights
            # Use sigmoid to keep weights positive and bounded
            alpha_weights = torch.sigmoid(self.class_alpha_weights[class_ids])
            
            # Reshape for broadcasting: (batch_size, 1)
            alpha_weights = alpha_weights.view(-1, 1)
            
            # Apply class-specific scaling
            scaling = base_scaling * alpha_weights
        else:
            # Use base scaling (inference or no class info)
            scaling = base_scaling
        
        return lora_output * scaling
    
    def get_class_group(self, class_id: int) -> str:
        """
        Get the group name (head/medium/tail) for a given class.
        
        Args:
            class_id: Class index
            
        Returns:
            Group name: 'head', 'medium', or 'tail'
        """
        if class_id in self.head_indices:
            return 'head'
        elif class_id in self.medium_indices:
            return 'medium'
        elif class_id in self.tail_indices:
            return 'tail'
        else:
            return 'unknown'
    
    def get_class_groups_info(self) -> Dict[str, List[int]]:
        """
        Get information about class groupings.
        
        Returns:
            Dictionary with head/medium/tail class indices
        """
        return {
            'head': self.head_indices.tolist(),
            'medium': self.medium_indices.tolist(),
            'tail': self.tail_indices.tolist()
        }
    
    def get_meta_parameters(self):
        """
        Return all meta parameters for optimization.
        
        Returns:
            List of parameters to be meta-optimized
        """
        meta_params = super().get_meta_parameters()
        
        # Add class-specific parameters
        if self.class_rank_weights is not None:
            meta_params.append(self.class_rank_weights)
        if self.class_alpha_weights is not None:
            meta_params.append(self.class_alpha_weights)
        
        return meta_params
    
    def get_effective_rank_weights(self) -> torch.Tensor:
        """
        Get the effective rank weights after sigmoid activation.
        
        Returns:
            Tensor of shape (num_classes,) with effective rank weights
        """
        if self.class_rank_weights is not None:
            return torch.sigmoid(self.class_rank_weights)
        else:
            return torch.ones(self.num_classes)
    
    def get_effective_alpha_weights(self) -> torch.Tensor:
        """
        Get the effective alpha weights after sigmoid activation.
        
        Returns:
            Tensor of shape (num_classes,) with effective alpha weights
        """
        if self.class_alpha_weights is not None:
            return torch.sigmoid(self.class_alpha_weights)
        else:
            return torch.ones(self.num_classes)
    
    def get_group_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about per-group parameter values.
        
        Returns:
            Dictionary with mean rank and alpha weights per group
        """
        rank_weights = self.get_effective_rank_weights()
        alpha_weights = self.get_effective_alpha_weights()
        
        stats = {}
        for group_name, indices in [
            ('head', self.head_indices),
            ('medium', self.medium_indices),
            ('tail', self.tail_indices)
        ]:
            if len(indices) > 0:
                stats[group_name] = {
                    'mean_rank_weight': rank_weights[indices].mean().item(),
                    'mean_alpha_weight': alpha_weights[indices].mean().item(),
                    'std_rank_weight': rank_weights[indices].std().item(),
                    'std_alpha_weight': alpha_weights[indices].std().item()
                }
        
        return stats


class ClassAwareMLPLoRA(ClassAwareMetaLoRA):
    """
    Class-aware LoRA for MLP layers with different output dimension.
    
    Extends ClassAwareMetaLoRA to support MLP layers where output dimension
    differs from input dimension.
    """
    
    def __init__(
        self,
        in_dim: int,
        bottle_dim: int,
        out_dim: int,
        alpha: float = 1.0,
        dtype=None,
        use_meta: bool = False,
        num_classes: int = 100,
        class_grouping: str = 'frequency',
        num_class_groups: int = 3,
        cls_num_list: Optional[np.ndarray] = None
    ):
        # Initialize parent
        super().__init__(
            in_dim=in_dim,
            bottle_dim=bottle_dim,
            alpha=alpha,
            dtype=dtype,
            use_meta=use_meta,
            num_classes=num_classes,
            class_grouping=class_grouping,
            num_class_groups=num_class_groups,
            cls_num_list=cls_num_list
        )
        
        # Replace lora_B with correct output dimension
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, out_dim, dtype=dtype))
        nn.init.zeros_(self.lora_B)
