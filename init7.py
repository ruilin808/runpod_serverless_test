#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 Optimized Qwen2-VL Fine-tuning Script for Table HTML Conversion
Enhanced with DeepSpeed ZeRO-3 for efficient GPU memory distribution
"""

import os
import torch
import gc
import json
from typing import Dict, Any

# Auto-detect and use all available GPUs
def setup_gpus():
    """Setup GPU configuration based on available devices"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s)")
        
        # Set memory management for better allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if gpu_count > 1:
            # Use all available GPUs
            gpu_ids = ','.join(str(i) for i in range(gpu_count))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            print(f"Using GPUs: {gpu_ids}")
            return gpu_count
        else:
            # Single GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("Using single GPU: 0")
            return 1
    else:
        print("No CUDA GPUs available")
        return 0

# Setup GPUs before importing other modules
gpu_count = setup_gpus()

from swift.llm import (
    get_model_tokenizer, get_template,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset as hf_load_dataset


def create_deepspeed_config(gpu_count: int) -> Dict[str, Any]:
    """Create DeepSpeed ZeRO-3 configuration optimized for multi-GPU setup"""
    
    # Base configuration for ZeRO-3
    config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "none",  # No CPU offloading as requested
                "pin_memory": False
            },
            "offload_param": {
                "device": "none",  # No CPU offloading as requested
                "pin_memory": False
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
        "bf16": {
            "enabled": "auto"
        },
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,  # Keep on GPU for speed
            "contiguous_memory_optimization": True,
            "number_checkpoints": 4,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        },
        "aio": {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True
        },
        "flops_profiler": {
            "enabled": False,
            "profile_step": 1,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": True,
            "output_file": None
        }
    }
    
    # Scale configuration based on GPU count
    if gpu_count >= 8:
        # High-end multi-GPU setup
        config["zero_optimization"]["reduce_bucket_size"] = 5e8
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e8
        config["zero_optimization"]["stage3_max_live_parameters"] = 3e9
        config["zero_optimization"]["stage3_max_reuse_distance"] = 3e9
        config["activation_checkpointing"]["number_checkpoints"] = 8
    elif gpu_count >= 4:
        # Medium multi-GPU setup
        config["zero_optimization"]["reduce_bucket_size"] = 2e8
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 2e8
        config["zero_optimization"]["stage3_max_live_parameters"] = 2e9
        config["zero_optimization"]["stage3_max_reuse_distance"] = 2e9
        config["activation_checkpointing"]["number_checkpoints"] = 6
    else:
        # Small multi-GPU setup
        config["zero_optimization"]["reduce_bucket_size"] = 1e8
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 1e8
        config["zero_optimization"]["stage3_max_live_parameters"] = 1e9
        config["zero_optimization"]["stage3_max_reuse_distance"] = 1e9
        config["activation_checkpointing"]["number_checkpoints"] = 4
    
    return config


def save_deepspeed_config(config: Dict[str, Any], output_dir: str) -> str:
    """Save DeepSpeed configuration to file"""
    config_path = os.path.join(output_dir, "deepspeed_config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def create_swift_format_single(sample):
    """Convert single sample to Swift format"""
    return {
        'messages': [
            {
                'role': 'user',
                'content': 'Write the HTML representation for this image of a medical table.'
            },
            {
                'role': 'assistant',
                'content': sample['html_table']
            }
        ],
        'images': [sample['image']]
    }


def filter_by_length(dataset, template, max_length):
    """Filter out samples that exceed max_length after encoding"""
    def is_valid_length(sample):
        try:
            encoded = template.encode(sample)
            return len(encoded['input_ids']) <= max_length
        except Exception:
            return False
    
    return dataset.filter(is_valid_length)


def truncate_html_content(sample, max_html_chars=None):
    """Truncate HTML content based on available GPU memory"""
    if max_html_chars is None:
        # With ZeRO-3, we can handle longer sequences due to memory partitioning
        if gpu_count >= 8:
            max_html_chars = 12000  # ZeRO-3 enables longer sequences
        elif gpu_count >= 4:
            max_html_chars = 10000
        else:
            max_html_chars = 8000
    
    if len(sample['html_table']) > max_html_chars:
        sample['html_table'] = sample['html_table'][:max_html_chars] + "..."
    return sample


def calculate_batch_size_and_accumulation(gpu_count, target_effective_batch_size=64):
    """Calculate optimal batch size and gradient accumulation steps for DeepSpeed ZeRO-3"""
    
    # With ZeRO-3, we can afford larger batch sizes due to memory partitioning
    if gpu_count >= 8:
        # With 8+ GPUs and ZeRO-3, we can use larger batch sizes
        per_device_batch_size = 4
        target_effective_batch_size = 128
    elif gpu_count >= 4:
        # With 4+ GPUs, moderate scaling
        per_device_batch_size = 3
        target_effective_batch_size = 96
    else:
        # 2-3 GPUs, still conservative but better than without ZeRO-3
        per_device_batch_size = 2
        target_effective_batch_size = 64
    
    # Calculate gradient accumulation to maintain effective batch size
    total_batch_size_per_step = per_device_batch_size * gpu_count
    gradient_accumulation_steps = max(1, target_effective_batch_size // total_batch_size_per_step)
    
    return per_device_batch_size, gradient_accumulation_steps


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main training function with DeepSpeed ZeRO-3"""
    
    logger = get_logger()
    seed_everything(42)
    
    # Clear initial GPU memory
    clear_gpu_memory()
    
    # Configuration - Enhanced for ZeRO-3
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
    output_dir = 'output'
    data_seed = 42
    
    # With ZeRO-3, we can handle longer sequences due to parameter partitioning
    if gpu_count >= 8:
        max_length = 16384  # ZeRO-3 enables longer sequences
    elif gpu_count >= 4:
        max_length = 14336
    else:
        max_length = 12288
    
    # LoRA configuration - Can be more aggressive with ZeRO-3
    if gpu_count >= 8:
        lora_rank = 16     # Higher rank possible with ZeRO-3
        lora_alpha = 64
    elif gpu_count >= 4:
        lora_rank = 12
        lora_alpha = 48
    else:
        lora_rank = 8
        lora_alpha = 32
    
    freeze_llm = False
    freeze_vit = True
    freeze_aligner = True
    
    # Calculate optimal batch size for ZeRO-3
    per_device_batch_size, gradient_accumulation_steps = calculate_batch_size_and_accumulation(
        gpu_count, target_effective_batch_size=64
    )
    
    # Create and save DeepSpeed configuration
    deepspeed_config = create_deepspeed_config(gpu_count)
    deepspeed_config_path = save_deepspeed_config(deepspeed_config, output_dir)
    
    logger.info(f"DeepSpeed ZeRO-3 Multi-GPU Configuration:")
    logger.info(f"  - GPUs available: {gpu_count}")
    logger.info(f"  - Per-device batch size: {per_device_batch_size}")
    logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {per_device_batch_size * gpu_count * gradient_accumulation_steps}")
    logger.info(f"  - Max sequence length: {max_length}")
    logger.info(f"  - LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    logger.info(f"  - DeepSpeed config saved to: {deepspeed_config_path}")
    
    # Adjust workers based on GPU count
    dataloader_workers = min(8, max(4, gpu_count))  # More workers with ZeRO-3
    
    # Training arguments - Enhanced for DeepSpeed ZeRO-3
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,  # Slightly higher LR with larger batch sizes
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_checkpointing=True,
        weight_decay=0.01,  # Reduced with ZeRO-3
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=200,
        eval_strategy='steps',
        eval_steps=200,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        metric_for_best_model='loss',
        save_total_limit=2,
        logging_steps=20,
        dataloader_num_workers=dataloader_workers,
        data_seed=data_seed,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        # DeepSpeed specific settings
        deepspeed=deepspeed_config_path,
        # Memory optimization settings
        dataloader_pin_memory=True,  # Can enable with ZeRO-3
        ddp_find_unused_parameters=False,
        ddp_timeout=1800,
        # Mixed precision handled by DeepSpeed
        bf16=True,  # BF16 often works better with ZeRO-3
        dataloader_prefetch_factor=4,  # Can increase with ZeRO-3
        save_safetensors=True,
        # Additional DeepSpeed optimizations
        ddp_backend="nccl",
        sharded_ddp="simple",  # Let DeepSpeed handle sharding
        prediction_loss_only=True,
    )
    
    # Load model with ZeRO-3 optimizations
    logger.info("Loading model with DeepSpeed ZeRO-3 optimizations...")
    model, processor = get_model_tokenizer(
        model_id_or_path,
        torch_dtype=torch.bfloat16,  # BF16 often works better with ZeRO-3
        device_map=None,  # Let DeepSpeed handle device placement
        low_cpu_mem_usage=True,
    )
    
    template = get_template(model.model_meta.template, processor, default_system=None, max_length=max_length)
    template.set_mode('train')
    template.model = model
    
    # Setup LoRA with enhanced settings for ZeRO-3
    target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                freeze_aligner=freeze_aligner)
    lora_config = LoraConfig(
        task_type='CAUSAL_LM', 
        r=lora_rank, 
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,  # Reduced dropout with ZeRO-3
        bias="none",
        use_rslora=True,  # Use rank-stabilized LoRA for better performance
        use_dora=False,  # Disable DoRA for memory efficiency
    )
    model = Swift.prepare_model(model, lora_config)
    
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')
    
    # Clear memory after model loading
    clear_gpu_memory()
    
    # Load and process dataset
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Apply HTML truncation with enhanced limits for ZeRO-3
    logger.info("Truncating HTML content...")
    train_processed = raw_dataset['train'].map(truncate_html_content)
    val_processed = raw_dataset['validation'].map(truncate_html_content)
    
    # Convert to Swift format
    train_processed = train_processed.map(create_swift_format_single)
    val_processed = val_processed.map(create_swift_format_single)
    
    logger.info(f"Original dataset sizes - Train: {len(train_processed)}, Val: {len(val_processed)}")
    
    # Filter out samples that are too long
    logger.info("Filtering samples by length...")
    train_processed = filter_by_length(train_processed, template, max_length)
    val_processed = filter_by_length(val_processed, template, max_length)
    
    logger.info(f"Filtered dataset sizes - Train: {len(train_processed)}, Val: {len(val_processed)}")
    
    # Convert to LazyLLMDataset
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=data_seed)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=data_seed)
    
    logger.info(f"Final dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)} samples")
    
    # Check if we have enough data after filtering
    if len(train_dataset) == 0:
        logger.error("No training samples remain after filtering! Consider increasing max_length or reducing HTML truncation.")
        return
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Train with DeepSpeed ZeRO-3
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        logger.info("Starting training with DeepSpeed ZeRO-3...")
        trainer.train()
        
        # Monitor GPU memory after training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory after training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        clear_gpu_memory()
        return
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Model saved to: {trainer.state.last_model_checkpoint}')
    
    # Final memory cleanup
    clear_gpu_memory()


if __name__ == "__main__":
    main()