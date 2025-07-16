#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script for Table HTML Conversion with Multi-GPU Support
"""

import os
import torch

# Auto-detect and use all available GPUs
def setup_gpus():
    """Setup GPU configuration based on available devices"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s)")
        
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


def truncate_html_content(sample, max_html_chars=8000):
    """Truncate HTML content if it's too long"""
    if len(sample['html_table']) > max_html_chars:
        sample['html_table'] = sample['html_table'][:max_html_chars] + "..."
    return sample


def calculate_batch_size_and_accumulation(gpu_count, base_batch_size=1, target_effective_batch_size=32):
    """Calculate optimal batch size and gradient accumulation steps for multi-GPU setup"""
    if gpu_count <= 1:
        return base_batch_size, target_effective_batch_size // base_batch_size
    
    # For multi-GPU, we can increase per-device batch size slightly
    # but keep it conservative for memory constraints
    per_device_batch_size = min(base_batch_size * 2, 4)  # Max 4 to avoid OOM
    
    # Calculate gradient accumulation to maintain effective batch size
    total_batch_size_per_step = per_device_batch_size * gpu_count
    gradient_accumulation_steps = max(1, target_effective_batch_size // total_batch_size_per_step)
    
    return per_device_batch_size, gradient_accumulation_steps


def main():
    """Main training function"""
    
    logger = get_logger()
    seed_everything(42)
    
    # Configuration - Increased max_length to accommodate longer sequences
    model_id_or_path = 'Qwen/Qwen2.5-VL-32B-Instruct'
    output_dir = 'output'
    data_seed = 42
    max_length = 20480  # Increased from 16384
    
    # LoRA configuration
    lora_rank = 8
    lora_alpha = 32
    freeze_llm = False
    freeze_vit = True
    freeze_aligner = True
    
    # Calculate optimal batch size and gradient accumulation for available GPUs
    per_device_batch_size, gradient_accumulation_steps = calculate_batch_size_and_accumulation(
        gpu_count, base_batch_size=1, target_effective_batch_size=32
    )
    
    logger.info(f"Multi-GPU Configuration:")
    logger.info(f"  - GPUs available: {gpu_count}")
    logger.info(f"  - Per-device batch size: {per_device_batch_size}")
    logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {per_device_batch_size * gpu_count * gradient_accumulation_steps}")
    
    # Training arguments - Adjusted for multi-GPU
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=50,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        metric_for_best_model='loss',
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=4 * gpu_count,  # Scale workers with GPU count
        data_seed=data_seed,
        remove_unused_columns=False,
        max_grad_norm=1.0,  # Added gradient clipping
        # Multi-GPU specific settings
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,  # Can help with performance
        ddp_timeout=1800,  # 30 minutes timeout for DDP
    )
    
    # Load model and setup template
    model, processor = get_model_tokenizer(model_id_or_path)
    template = get_template(model.model_meta.template, processor, default_system=None, max_length=max_length)
    template.set_mode('train')
    template.model = model
    
    # Setup LoRA
    target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                freeze_aligner=freeze_aligner)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')
    
    # Load and process dataset
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Apply HTML truncation first (optional - use if you want to truncate rather than filter)
    # train_processed = raw_dataset['train'].map(truncate_html_content)
    # val_processed = raw_dataset['validation'].map(truncate_html_content)
    
    # Convert to Swift format
    train_processed = raw_dataset['train'].map(create_swift_format_single)
    val_processed = raw_dataset['validation'].map(create_swift_format_single)
    
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
        logger.error("No training samples remain after filtering! Consider increasing max_length or truncating HTML content.")
        return
    
    # Train
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
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Model saved to: {trainer.state.last_model_checkpoint}')


if __name__ == "__main__":
    main()