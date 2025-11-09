from omegaconf import OmegaConf

# Create base config dictionary
config = {
    "dataset": "",  # Dataset name
    "root": "",  # Directory where datasets are stored
    "imb_factor": None,  # for long-tailed cifar dataset
    "head_init_folder": None,  # path to class mean file
    "backbone": "",
    "resolution": 224,
    "output_dir": None,  # Directory to save output files
    "print_freq": 10,  # How often (batch) to print training information
    "seed": None,  # use manual seed
    "deterministic": False,  # output deterministic results
    "num_workers": 20,
    "prec": "fp16",  # fp16, fp32, amp
    "num_epochs": 10,
    "batch_size": 128,
    "accum_step": 1,  # for gradient accumulation
    "lr": 0.01,
    "scheduler": "CosineAnnealingLR",
    "weight_decay": 5e-4,
    "momentum": 0.9,
    "loss_type": "LA",  # "CE" / "Focal" / "LDAM" / "CB" / "GRW" / "BS" / "LA" / "LADE"
    "classifier": "CosineClassifier",
    "scale": 25,  # for cosine classifier
    # Fine-tuning options
    "fine_tuning": False,
    "head_only": False,
    "full_tuning": False,  # WARN: full_tuning is not implemented in current code.
    "bias_tuning": False,
    "ln_tuning": False,
    "bn_tuning": False,
    "vpt_shallow": False,
    "vpt_deep": False,
    "adapter": False,
    "adaptformer": False,
    "lora": False,
    "lora_mlp": False,
    "scale_alpha": 1,
    "ssf_attn": False,
    "ssf_mlp": False,
    "ssf_ln": False,
    "mask": False,
    "partial": None,
    "vpt_len": None,
    "adapter_dim": None,
    "adaptformer_scale": "learnable",
    "mask_ratio": None,
    "mask_seed": None,
    "init_head": None,
    "prompt": "default",
    "tte": False,
    "expand": 24,
    "tte_mode": "fivecrop",
    "randaug_times": 1,
    "zero_shot": False,
    "test_only": False,
    "test_train": False,
    "model_dir": None,
    # FLoRA configs
    "use_flora": False,
    "flora": {
        "arch": {
            "modules": ["q", "v", "k", "out", "mlp1", "mlp2"],
            "layers": list(range(12)),
            "rank": 4,
            "alpha": None,  # 添加 alpha 配置，默认为 None
        },
        "optimizer": {
            "default_lr": 2e-2,
            "lr_config": {
                "layers": {
                    # "0-2": 3e-2,
                    # "9-11": 1e-2,
                },
                "modules": {
                    # "q": 1e-2,
                    # "v": 2e-2,
                },
                "specific": {
                    # "0.q": 4e-2,
                },
            },
        },
    },
    "use_meta": False,
    "meta_data_ratio": 0.1,
    "meta_lr": 1e-3,
    "meta_update_freq": 1,
    "meta_inner_steps": 5,
    # Class-aware meta-learning configs
    "use_class_aware": False,
    "meta_objective": "balanced_accuracy",  # balanced_accuracy, gmean, worst_case
    "focus_on_tail": True,
    "tail_loss_weight": 2.0,
    "rank_divergence_penalty": 0.01,
    "alpha_smoothness_penalty": 0.005,
    "head_threshold": 100,
    "tail_threshold": 20,
    "head_rank_factor": 0.5,
    "tail_rank_factor": 2.0,
    "head_alpha_factor": 0.5,
    "tail_alpha_factor": 2.0,
}

# Convert to OmegaConf
cfg = OmegaConf.create(config)
