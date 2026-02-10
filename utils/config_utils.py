"""
Configuration utilities for LBM training.

This module provides utilities to load and merge configuration from YAML files
with command line arguments, making it easier to manage training configurations.
"""

import argparse
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def flatten_config(config: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary.
    
    Example:
        {'model': {'learning_rate': 0.001}} -> {'model_learning_rate': 0.001}
    """
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_config(config: Dict[str, Any], sep: str = '_') -> Dict[str, Any]:
    """
    Unflatten configuration dictionary.
    
    Example:
        {'model_learning_rate': 0.001} -> {'model': {'learning_rate': 0.001}}
    """
    result = {}
    for key, value in config.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries recursively.
    Values in override_config take precedence.
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def config_to_args(config: Dict[str, Any]) -> argparse.Namespace:
    """Convert configuration dictionary to argparse.Namespace."""
    flattened = flatten_config(config)
    
    # Convert keys to match argument parser format
    args_dict = {}
    for key, value in flattened.items():
        # Convert nested keys to argument format
        arg_key = key.replace('_', '_')
        args_dict[arg_key] = value
    
    return argparse.Namespace(**args_dict)


def update_args_from_config(args: argparse.Namespace, config_path: Optional[str] = None) -> argparse.Namespace:
    """
    Update argparse.Namespace with values from configuration file.
    Command line arguments take precedence over config file values.
    """
    if config_path is None:
        return args
    
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found. Using command line arguments only.")
        return args
    
    # Load config from file
    file_config = load_config(config_path)
    
    # Convert args to dict, excluding None values (not set via command line)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    # Flatten file config for easier merging
    flat_file_config = flatten_config(file_config)
    
    # Map config keys to argument names
    config_to_arg_mapping = {
        # Model config
        'model_pretrained_model_name_or_path': 'pretrained_model_name_or_path',
        'model_revision': 'revision',
        'model_variant': 'variant',
        
        # Data config
        'data_train_data_dir': 'train_data_dir',
        'data_source_key': 'source_key',
        'data_target_key': 'target_key',
        'data_mask_key': 'mask_key',
        'data_resolution': 'resolution',
        
        # Training config
        'training_output_dir': 'output_dir',
        'training_seed': 'seed',
        'training_train_batch_size': 'train_batch_size',
        'training_num_train_epochs': 'num_train_epochs',
        'training_max_train_steps': 'max_train_steps',
        'training_gradient_accumulation_steps': 'gradient_accumulation_steps',
        'training_gradient_checkpointing': 'gradient_checkpointing',
        'training_learning_rate': 'learning_rate',
        'training_dataloader_num_workers': 'dataloader_num_workers',
        
        # LBM config
        'lbm_bridge_noise_sigma': 'bridge_noise_sigma',
        'lbm_latent_loss_weight': 'latent_loss_weight',
        'lbm_latent_loss_type': 'latent_loss_type',
        'lbm_pixel_loss_weight': 'pixel_loss_weight',
        'lbm_pixel_loss_type': 'pixel_loss_type',
        'lbm_pixel_loss_max_size': 'pixel_loss_max_size',
        'lbm_timestep_sampling': 'timestep_sampling',
        'lbm_logit_mean': 'logit_mean',
        'lbm_logit_std': 'logit_std',
        'lbm_selected_timesteps': 'selected_timesteps',
        'lbm_timestep_probs': 'timestep_probs',
        
        # Optimizer config
        'optimizer_type': 'optimizer',
        'optimizer_adam_beta1': 'adam_beta1',
        'optimizer_adam_beta2': 'adam_beta2',
        'optimizer_adam_weight_decay': 'adam_weight_decay',
        'optimizer_adam_epsilon': 'adam_epsilon',
        'optimizer_max_grad_norm': 'max_grad_norm',
        
        # Scheduler config
        'scheduler_lr_scheduler': 'lr_scheduler',
        'scheduler_lr_warmup_steps': 'lr_warmup_steps',
        
        # Logging config
        'logging_checkpointing_steps': 'checkpointing_steps',
        'logging_checkpoints_total_limit': 'checkpoints_total_limit',
        'logging_resume_from_checkpoint': 'resume_from_checkpoint',
        'logging_logging_dir': 'logging_dir',
        'logging_report_to': 'report_to',
        
        # Hardware config
        'hardware_mixed_precision': 'mixed_precision',
        'hardware_allow_tf32': 'allow_tf32',
    }
    
    # Update args with config values (only if not set via command line)
    for config_key, arg_name in config_to_arg_mapping.items():
        if config_key in flat_file_config and arg_name not in args_dict:
            setattr(args, arg_name, flat_file_config[config_key])
    
    return args


def add_config_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --config argument to argument parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. Command line arguments override config file values.",
    )
    return parser


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def print_config(config: Dict[str, Any], title: str = "Configuration") -> None:
    """Pretty print configuration."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    def _print_dict(d: Dict[str, Any], indent: int = 0) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}{key}:")
                _print_dict(value, indent + 1)
            else:
                print(f"{'  ' * indent}{key}: {value}")
    
    _print_dict(config)
    print(f"{'='*50}\n")


# Example usage
if __name__ == "__main__":
    # Example of loading and using config
    config_path = "configs/lbm_training_config.yaml"
    
    if Path(config_path).exists():
        config = load_config(config_path)
        print_config(config, "Loaded Configuration")
        
        # Example of flattening and unflattening
        flat_config = flatten_config(config)
        print("Flattened keys:", list(flat_config.keys())[:5])
        
        unflat_config = unflatten_config(flat_config)
        print("Unflattened structure matches original:", config == unflat_config)
    else:
        print(f"Config file {config_path} not found.")
