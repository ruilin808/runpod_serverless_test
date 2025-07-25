#!/usr/bin/env python3
"""
Image Token-Aware Token Clamping Fix for Qwen2-VL
Preserves special tokens while fixing out-of-vocabulary issues
"""

import os
import torch
import gc

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


def get_special_token_ranges(tokenizer):
    """Identify ranges of special tokens that should NOT be clamped"""
    special_tokens = set()
    vocab_size = tokenizer.vocab_size
    
    # Add known special tokens
    for attr in ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id']:
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_tokens.add(token_id)
    
    # For Qwen2-VL, image tokens are typically in a specific range beyond vocab_size
    # Based on your logs, tokens 151655 appear, which suggests image tokens start around 151643
    # Let's preserve tokens in the range [151643, 151700] as potential image tokens
    image_token_start = vocab_size  # 151643
    image_token_end = vocab_size + 100  # Give some buffer
    
    print(f"Preserving special tokens: {special_tokens}")
    print(f"Preserving image token range: [{image_token_start}, {image_token_end})")
    
    return special_tokens, (image_token_start, image_token_end)


def smart_clamp_tokens(token_ids, vocab_size, special_tokens, image_token_range, replacement_token=0):
    """Intelligently clamp tokens while preserving special tokens"""
    if isinstance(token_ids, list):
        token_ids = torch.tensor(token_ids)
    
    # Find tokens that are out of regular vocabulary but NOT special tokens
    invalid_mask = token_ids >= vocab_size
    
    if invalid_mask.any():
        # Don't clamp special tokens or image tokens
        image_start, image_end = image_token_range
        
        # Create mask for tokens that should be preserved
        preserve_mask = torch.zeros_like(invalid_mask, dtype=torch.bool)
        
        for special_token in special_tokens:
            preserve_mask |= (token_ids == special_token)
            
        # Preserve image tokens in the expected range
        preserve_mask |= ((token_ids >= image_start) & (token_ids < image_end))
        
        # Only clamp tokens that are invalid AND not preserved
        clamp_mask = invalid_mask & (~preserve_mask)
        
        if clamp_mask.any():
            token_ids = token_ids.clone()
            token_ids[clamp_mask] = replacement_token
            
            clamped_count = clamp_mask.sum().item()
            preserved_count = (invalid_mask & preserve_mask).sum().item()
            print(f"Clamped {clamped_count} out-of-vocab tokens, preserved {preserved_count} special tokens")
        else:
            preserved_count = invalid_mask.sum().item()
            print(f"Preserved {preserved_count} special/image tokens")
    
    return token_ids


def encode_sample_with_smart_clamping(template, sample, tokenizer, special_tokens, image_token_range):
    """Encode sample with intelligent token clamping"""
    try:
        # Get the raw encoding first
        encoded = template.encode(sample)
        vocab_size = tokenizer.vocab_size
        
        # Handle both list and tensor cases
        input_ids = encoded['input_ids']
        labels = encoded.get('labels', None)
        
        # Convert lists to tensors if needed
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        if isinstance(labels, list):
            labels = torch.tensor(labels)
            
        # Smart clamp input_ids
        input_ids = smart_clamp_tokens(
            input_ids, vocab_size, special_tokens, image_token_range, 
            replacement_token=tokenizer.unk_token_id or 0
        )
        
        # Smart clamp labels (but preserve -100 ignore tokens)
        if labels is not None:
            # Only clamp non-ignore tokens
            non_ignore_mask = labels != -100
            if non_ignore_mask.any():
                valid_labels = labels[non_ignore_mask]
                clamped_labels = smart_clamp_tokens(
                    valid_labels, vocab_size, special_tokens, image_token_range,
                    replacement_token=tokenizer.unk_token_id or 0
                )
                labels = labels.clone()
                labels[non_ignore_mask] = clamped_labels
        
        # Update the encoded dict
        encoded['input_ids'] = input_ids
        if labels is not None:
            encoded['labels'] = labels
            
        return encoded, None
        
    except Exception as e:
        return None, f"Encoding failed: {str(e)}"


def filter_by_length(dataset, template, tokenizer, special_tokens, image_token_range, max_length):
    """Filter out samples that exceed max_length after encoding"""
    valid_samples = []
    invalid_count = 0
    
    print(f"Processing {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        # Try to encode with smart clamping
        encoded, error = encode_sample_with_smart_clamping(
            template, sample, tokenizer, special_tokens, image_token_range
        )
        
        if encoded is None:
            invalid_count += 1
            if invalid_count <= 5:  # Show first 5 errors
                print(f"Sample {i} failed encoding: {error}")
            continue
            
        # Check sequence length
        if len(encoded['input_ids']) > max_length:
            invalid_count += 1
            continue
            
        valid_samples.append(sample)
        
        # Progress update
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples, {len(valid_samples)} valid")
    
    print(f"Filtering complete: {len(valid_samples)} valid, {invalid_count} invalid")
    
    # Convert back to dataset format
    from datasets import Dataset
    return Dataset.from_list(valid_samples)


class SmartTokenClampingDataCollator:
    """Data collator that intelligently clamps out-of-vocabulary tokens"""
    
    def __init__(self, base_collator, vocab_size, special_tokens, image_token_range, replacement_token=0):
        self.base_collator = base_collator
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.image_token_range = image_token_range
        self.replacement_token = replacement_token
    
    def __call__(self, features):
        # First apply the base collator
        batch = self.base_collator(features)
        
        # Then intelligently clamp any out-of-vocab tokens
        for key in ['input_ids', 'labels']:
            if key in batch:
                value = batch[key]
                
                # Convert lists to tensors if needed
                if isinstance(value, list):
                    value = torch.tensor(value)
                    
                # Smart clamp out-of-vocab tokens
                if key == 'labels':
                    # For labels, only clamp non-ignore tokens
                    non_ignore_mask = value != -100
                    if non_ignore_mask.any():
                        clamped_value = value.clone()
                        valid_tokens = value[non_ignore_mask]
                        clamped_tokens = smart_clamp_tokens(
                            valid_tokens, self.vocab_size, self.special_tokens, 
                            self.image_token_range, self.replacement_token
                        )
                        clamped_value[non_ignore_mask] = clamped_tokens
                        batch[key] = clamped_value
                    else:
                        batch[key] = value
                else:
                    # For input_ids, clamp all invalid tokens
                    batch[key] = smart_clamp_tokens(
                        value, self.vocab_size, self.special_tokens, 
                        self.image_token_range, self.replacement_token
                    )
        
        return batch


def calculate_batch_size_and_accumulation(gpu_count, base_batch_size=1, target_effective_batch_size=32):
    """Calculate optimal batch size and gradient accumulation steps for multi-GPU setup"""
    if gpu_count <= 1:
        return base_batch_size, target_effective_batch_size // base_batch_size
    
    # Scale batch size based on GPU count while being conservative about memory
    if gpu_count >= 8:
        per_device_batch_size = 2
        target_effective_batch_size = 64
    elif gpu_count >= 4:
        per_device_batch_size = 1
        target_effective_batch_size = 32
    else:
        per_device_batch_size = 1
        target_effective_batch_size = 16
    
    total_batch_size_per_step = per_device_batch_size * gpu_count
    gradient_accumulation_steps = max(1, target_effective_batch_size // total_batch_size_per_step)
    
    return per_device_batch_size, gradient_accumulation_steps


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main training function"""
    
    # Enable CUDA debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    logger = get_logger()
    seed_everything(42)
    
    # Clear initial GPU memory
    clear_gpu_memory()
    
    # Configuration
    model_id_or_path = 'Qwen/Qwen2.5-VL-32B-Instruct'
    output_dir = 'output'
    data_seed = 42
    
    # Scale max_length based on GPU count for better memory distribution
    if gpu_count >= 8:
        max_length = 12288
    elif gpu_count >= 4:
        max_length = 10240
    else:
        max_length = 8192
    
    # LoRA configuration
    if gpu_count >= 8:
        lora_rank = 8
        lora_alpha = 32
    elif gpu_count >= 4:
        lora_rank = 6
        lora_alpha = 24
    else:
        lora_rank = 4
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
    dataloader_workers = min(4, max(2, gpu_count // 2))
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        warmup_steps=10,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=100,
        eval_strategy='steps',
        eval_steps=100,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        metric_for_best_model='loss',
        save_total_limit=3,
        logging_steps=10,
        dataloader_num_workers=dataloader_workers,
        data_seed=data_seed,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        # Memory optimization settings
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        ddp_timeout=1800,
        # Mixed precision training
        fp16=True,
        dataloader_prefetch_factor=2,
        ddp_bucket_cap_mb=25,
        save_safetensors=True,
        # Optimizer settings
        lr_scheduler_kwargs={},
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
    )

    # Load model with memory optimizations
    logger.info("Loading model with memory optimizations...")
    model, processor = get_model_tokenizer(
        model_id_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    
    # Get tokenizer for debugging and validation
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"UNK token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
    
    # Get special token information
    special_tokens, image_token_range = get_special_token_ranges(tokenizer)
    
    template = get_template(model.model_meta.template, processor, default_system=None, max_length=max_length)
    template.set_mode('train')
    template.model = model
    
    # Setup LoRA
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
    
    # Check model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Clear memory after model loading
    clear_gpu_memory()
    
    # Load and process dataset
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Convert raw dataset to Swift format (NO TRUNCATION)
    logger.info("Converting to Swift format...")
    train_processed = raw_dataset['train'].map(create_swift_format_single)
    val_processed = raw_dataset['validation'].map(create_swift_format_single)
    
    logger.info(f"Original dataset sizes - Train: {len(train_processed)}, Val: {len(val_processed)}")
    
    # Filter with smart token clamping
    logger.info("Filtering samples with smart token clamping...")
    train_processed = filter_by_length(train_processed, template, tokenizer, special_tokens, image_token_range, max_length)
    val_processed = filter_by_length(val_processed, template, tokenizer, special_tokens, image_token_range, max_length)
    
    logger.info(f"Filtered dataset sizes - Train: {len(train_processed)}, Val: {len(val_processed)}")
    
    # Check if we have enough data after filtering
    if len(train_processed) == 0:
        logger.error("No training samples remain after filtering!")
        return
    
    # Create custom encoding function that clamps tokens smartly
    def encode_with_smart_clamping(sample):
        encoded, error = encode_sample_with_smart_clamping(
            template, sample, tokenizer, special_tokens, image_token_range
        )
        if encoded is None:
            raise ValueError(f"Failed to encode sample: {error}")
        return encoded
    
    # Convert to LazyLLMDataset with smart clamping
    train_dataset = LazyLLMDataset(train_processed, encode_with_smart_clamping, random_state=data_seed)
    val_dataset = LazyLLMDataset(val_processed, encode_with_smart_clamping, random_state=data_seed)
    
    logger.info(f"Final dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)} samples")
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Create smart token-clamping data collator
    base_collator = template.data_collator
    clamping_collator = SmartTokenClampingDataCollator(
        base_collator, 
        tokenizer.vocab_size, 
        special_tokens,
        image_token_range,
        replacement_token=tokenizer.unk_token_id or 0
    )
    
    # Train
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=clamping_collator,  # Use smart clamping collator
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        # Test first batch
        logger.info("Testing first training batch...")
        train_dataloader = trainer.get_train_dataloader()
        first_batch = next(iter(train_dataloader))
        
        # Check the batch for any remaining invalid tokens
        for key in ['input_ids', 'labels']:
            if key in first_batch:
                tensor = first_batch[key]
                if key == 'labels':
                    valid_mask = tensor != -100
                    if valid_mask.any():
                        valid_tokens = tensor[valid_mask]
                        max_val = valid_tokens.max().item()
                        if max_val >= tokenizer.vocab_size:
                            # Check if it's in the preserved image token range
                            image_start, image_end = image_token_range
                            if max_val < image_start or max_val >= image_end:
                                raise ValueError(f"Invalid {key} found in batch: max={max_val}, not in preserved range [{image_start}, {image_end})")
                else:
                    max_val = tensor.max().item()
                    if max_val >= tokenizer.vocab_size:
                        # Check if it's in the preserved image token range
                        image_start, image_end = image_token_range
                        if max_val < image_start or max_val >= image_end:
                            raise ValueError(f"Invalid {key} found in batch: max={max_val}, not in preserved range [{image_start}, {image_end})")
        
        # Test forward pass
        first_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in first_batch.items()}
        
        with torch.no_grad():
            outputs = model(**first_batch)
            logger.info(f"First batch test successful. Loss: {outputs.loss}")
        
        # If first batch works, proceed with training
        trainer.train()
        
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