#!/usr/bin/env python3
"""
Memory-Optimized Qwen2-VL Fine-tuning Script with TEDS Evaluation
"""

import os
import torch
import gc
from typing import Dict, List
import numpy as np

# Set GPU config before any CUDA operations
def setup_gpus():
    """Setup GPU configuration based on available devices"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s)")
        
        # Set memory management for better allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        return gpu_count
    else:
        print("No CUDA GPUs available")
        return 0

# Setup GPUs early but don't set CUDA_VISIBLE_DEVICES
gpu_count = setup_gpus()

from swift.llm import (
    get_model_tokenizer, get_template,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset as hf_load_dataset
from table_recognition_metric import TEDS


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
    dropped_count = 0
    
    def is_valid_length(sample):
        nonlocal dropped_count
        try:
            encoded = template.encode(sample)
            return len(encoded['input_ids']) <= max_length
        except Exception as e:
            print(f"Encoding failed for sample: {e}")
            dropped_count += 1
            return False
    
    filtered_dataset = dataset.filter(is_valid_length)
    if dropped_count > 0:
        print(f"Dropped {dropped_count} samples due to encoding errors")
    
    return filtered_dataset


def calculate_batch_size_and_accumulation(gpu_count, base_batch_size=1, target_effective_batch_size=32):
    """Calculate optimal batch size and gradient accumulation steps for multi-GPU setup"""
    if gpu_count <= 1:
        return base_batch_size, target_effective_batch_size // base_batch_size
    
    # Scale batch size based on GPU count while being conservative about memory
    if gpu_count >= 8:
        # With 8+ GPUs, we can afford slightly larger per-device batch size
        per_device_batch_size = 2
        target_effective_batch_size = 64  # Scale up effective batch size
    elif gpu_count >= 4:
        # With 4+ GPUs, moderate scaling
        per_device_batch_size = 1
        target_effective_batch_size = 32
    else:
        # 2-3 GPUs, conservative approach
        per_device_batch_size = 1
        target_effective_batch_size = 16
    
    # Calculate gradient accumulation to maintain effective batch size
    total_batch_size_per_step = per_device_batch_size * gpu_count
    gradient_accumulation_steps = max(1, target_effective_batch_size // total_batch_size_per_step)
    
    return per_device_batch_size, gradient_accumulation_steps


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def compute_teds_metrics(model, processor, template, val_dataset, device, max_samples=50):
    """Compute TEDS metrics on validation set"""
    model.eval()
    teds = TEDS()
    scores = []
    
    # Sample subset for evaluation to avoid slowdown
    eval_samples = min(max_samples, len(val_dataset))
    
    with torch.no_grad():
        for i in range(eval_samples):
            try:
                # Get sample
                sample = val_dataset[i]
                
                # Prepare input for generation
                messages = [{'role': 'user', 'content': 'Write the HTML representation for this image of a medical table.'}]
                inputs = processor.apply_chat_template(messages, images=[sample['images'][0]], add_generation_prompt=True)
                
                # Generate prediction
                inputs = inputs.to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
                
                # Decode prediction
                generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract HTML (assumes it's at the end after the prompt)
                pred_html = generated_text.split('Write the HTML representation for this image of a medical table.')[-1].strip()
                
                # Get ground truth HTML
                gt_html = sample['messages'][1]['content']
                
                # Compute TEDS score
                score = teds(gt_html, pred_html)
                scores.append(score)
                
            except Exception as e:
                print(f"Error computing TEDS for sample {i}: {e}")
                scores.append(0.0)  # Add 0 for failed samples
    
    model.train()
    return {
        'teds_score': np.mean(scores),
        'teds_std': np.std(scores),
        'eval_samples': eval_samples
    }


class CustomTrainer(Seq2SeqTrainer):
    """Custom trainer with TEDS evaluation"""
    
    def __init__(self, *args, **kwargs):
        self.val_dataset_raw = kwargs.pop('val_dataset_raw', None)
        super().__init__(*args, **kwargs)
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include TEDS metrics"""
        # Run standard evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add TEDS evaluation
        if self.val_dataset_raw is not None:
            try:
                teds_metrics = compute_teds_metrics(
                    self.model, 
                    self.tokenizer, 
                    self.template,
                    self.val_dataset_raw,
                    self.args.device,
                    max_samples=20  # Evaluate on subset to avoid slowdown
                )
                
                # Add TEDS metrics to results
                metrics.update({
                    f'{metric_key_prefix}_teds_score': teds_metrics['teds_score'],
                    f'{metric_key_prefix}_teds_std': teds_metrics['teds_std'],
                    f'{metric_key_prefix}_teds_samples': teds_metrics['eval_samples']
                })
                
                self.log(metrics)
                
            except Exception as e:
                print(f"TEDS evaluation failed: {e}")
        
        return metrics


def main():
    """Main training function"""
    
    logger = get_logger()
    seed_everything(42)
    
    # Clear initial GPU memory
    clear_gpu_memory()
    
    # Configuration - Scale parameters based on GPU count
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
    output_dir = 'output'
    data_seed = 42
    
    # Scale max_length based on GPU count for better memory distribution
    if gpu_count >= 8:
        max_length = 12288  # Can handle longer sequences with more GPUs
    elif gpu_count >= 4:
        max_length = 10240  # Moderate scaling
    else:
        max_length = 8192   # Conservative for 2-3 GPUs
    
    # LoRA configuration - Scale based on GPU count
    if gpu_count >= 8:
        lora_rank = 8      # Can afford higher rank with more GPUs
        lora_alpha = 32
    elif gpu_count >= 4:
        lora_rank = 6      # Moderate scaling
        lora_alpha = 24
    else:
        lora_rank = 4      # Conservative for fewer GPUs
        lora_alpha = 16
    
    freeze_llm = False
    freeze_vit = True
    freeze_aligner = True
    
    # Calculate optimal batch size and gradient accumulation for available GPUs
    per_device_batch_size, gradient_accumulation_steps = calculate_batch_size_and_accumulation(
        gpu_count, base_batch_size=1, target_effective_batch_size=32
    )
    
    logger.info(f"Multi-GPU Scaling Configuration:")
    logger.info(f"  - GPUs available: {gpu_count}")
    logger.info(f"  - Per-device batch size: {per_device_batch_size}")
    logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {per_device_batch_size * gpu_count * gradient_accumulation_steps}")
    logger.info(f"  - Max sequence length: {max_length}")
    logger.info(f"  - LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    
    # Adjust workers based on GPU count
    dataloader_workers = min(4, max(2, gpu_count // 2))  # Scale workers with GPUs but cap at 4
    
    # Training arguments - Memory optimized
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
        save_steps=100,
        eval_strategy='steps',
        eval_steps=5,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        metric_for_best_model='eval_teds_score',  # Use TEDS as best model metric
        greater_is_better=True,  # Higher TEDS score is better
        save_total_limit=3,
        logging_steps=10,
        dataloader_num_workers=dataloader_workers,
        data_seed=data_seed,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        # Memory optimization settings
        dataloader_pin_memory=True,  # Enable for GPU training
        ddp_find_unused_parameters=False,
        ddp_timeout=1800,
        # Use bf16 for better stability
        bf16=True,
        dataloader_prefetch_factor=2,
        ddp_bucket_cap_mb=25,
        save_safetensors=True,
    )
    
    # Load model with memory optimizations
    logger.info("Loading model with memory optimizations...")
    model, processor = get_model_tokenizer(
        model_id_or_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for consistency
        device_map='auto',  # Let accelerate handle device placement
        low_cpu_mem_usage=True,
    )
    
    template = get_template(model.model_meta.template, processor, default_system=None, max_length=max_length)
    template.set_mode('train')
    template.model = model
    
    # Setup LoRA with memory-efficient settings
    target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                freeze_aligner=freeze_aligner)
    lora_config = LoraConfig(
        task_type='CAUSAL_LM', 
        r=lora_rank, 
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    model = Swift.prepare_model(model, lora_config)
    
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')
    
    # Clear memory after model loading
    clear_gpu_memory()
    
    # Load and process dataset
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Use raw dataset directly without truncation
    train_processed = raw_dataset['train']
    val_processed = raw_dataset['validation']
    
    # Convert to Swift format
    train_processed = train_processed.map(create_swift_format_single, desc="Converting train to Swift format")
    val_processed = val_processed.map(create_swift_format_single, desc="Converting validation to Swift format")
    
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
        logger.error("No training samples remain after filtering! Consider increasing max_length.")
        return
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Train with custom trainer that includes TEDS evaluation
    model.enable_input_require_grads()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        val_dataset_raw=val_processed,  # Pass raw dataset for TEDS evaluation
    )
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Clear memory on failure
        clear_gpu_memory()
        raise  # Re-raise to see full traceback
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Model saved to: {trainer.state.last_model_checkpoint}')
    
    # Final memory cleanup
    clear_gpu_memory()


if __name__ == "__main__":
    main()