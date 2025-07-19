#!/usr/bin/env python3
"""
Memory-Optimized Qwen2-VL Fine-tuning Script for Table HTML Conversion with Multi-GPU Support
FIXED VERSION WITHOUT TRUNCATION - PRESERVES FULL HTML CONTENT
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


def validate_and_fix_sample_encoding(sample, template, tokenizer):
    """Validate and attempt to fix sample encoding issues"""
    try:
        # Try to encode the sample
        encoded = template.encode(sample)
        
        # Fix list-to-tensor conversion issue that affects 32B model
        input_ids = encoded['input_ids']
        labels = encoded.get('labels', None)
        
        # Convert lists to tensors if needed
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
            encoded['input_ids'] = input_ids
            
        if isinstance(labels, list):
            labels = torch.tensor(labels)
            encoded['labels'] = labels
        
        # Check for token validity
        vocab_size = tokenizer.vocab_size
        
        # Check input_ids
        if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
            print(f"Invalid input_ids found: min={input_ids.min()}, max={input_ids.max()}, vocab_size={vocab_size}")
            return None, f"Invalid input_ids: range [{input_ids.min()}, {input_ids.max()}] vs vocab_size {vocab_size}"
            
        # Check labels if present
        if labels is not None:
            valid_mask = labels != -100
            if valid_mask.any():
                valid_labels = labels[valid_mask]
                if torch.any(valid_labels >= vocab_size) or torch.any(valid_labels < 0):
                    print(f"Invalid labels found: min={valid_labels.min()}, max={valid_labels.max()}, vocab_size={vocab_size}")
                    return None, f"Invalid labels: range [{valid_labels.min()}, {valid_labels.max()}] vs vocab_size {vocab_size}"
        
        return encoded, None
        
    except Exception as e:
        return None, f"Encoding failed: {str(e)}"


def safe_filter_dataset(dataset, template, tokenizer, max_length):
    """Safely filter dataset by encoding validation and length"""
    valid_samples = []
    invalid_count = 0
    encoding_errors = {}
    
    print(f"Filtering {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        # Check basic HTML content (but don't truncate)
        html_content = sample['html_table']
        
        # Skip completely empty samples
        if not html_content or len(html_content.strip()) == 0:
            invalid_count += 1
            continue
            
        # Try to encode and validate
        encoded, error = validate_and_fix_sample_encoding(sample, template, tokenizer)
        
        if encoded is None:
            invalid_count += 1
            error_type = error.split(':')[0] if ':' in error else error
            encoding_errors[error_type] = encoding_errors.get(error_type, 0) + 1
            if invalid_count <= 5:  # Show first 5 errors for debugging
                print(f"Sample {i} invalid: {error}")
            continue
            
        # Check sequence length
        if len(encoded['input_ids']) > max_length:
            invalid_count += 1
            continue
            
        valid_samples.append(sample)
        
        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples, {len(valid_samples)} valid so far")
    
    print(f"\nFiltering complete:")
    print(f"- Valid samples: {len(valid_samples)}")
    print(f"- Invalid samples: {invalid_count}")
    if encoding_errors:
        print(f"- Encoding error breakdown: {encoding_errors}")
    
    # Convert back to dataset format
    from datasets import Dataset
    return Dataset.from_list(valid_samples)


def debug_tokenizer_and_vocab(tokenizer, model):
    """Debug tokenizer and vocabulary information"""
    print(f"\n=== TOKENIZER DEBUG INFO ===")
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Check for special tokens
    special_tokens = {}
    for attr in ['pad_token', 'eos_token', 'bos_token', 'unk_token']:
        token = getattr(tokenizer, attr, None)
        if token:
            token_id = getattr(tokenizer, f'{attr}_id', None)
            special_tokens[attr] = (token, token_id)
    
    print(f"Special tokens: {special_tokens}")
    
    # Test basic HTML encoding
    test_html = "<table><tr><td>Test</td></tr></table>"
    try:
        test_encoded = tokenizer.encode(test_html)
        print(f"Test HTML encoding successful: {len(test_encoded)} tokens")
        print(f"Token range: [{min(test_encoded)}, {max(test_encoded)}]")
    except Exception as e:
        print(f"Test HTML encoding failed: {e}")
    
    print("=== END TOKENIZER DEBUG ===\n")


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


class FixedDataCollator:
    """Custom data collator that handles list-to-tensor conversion for 32B model"""
    
    def __init__(self, base_collator):
        self.base_collator = base_collator
    
    def __call__(self, features):
        # First apply the base collator
        batch = self.base_collator(features)
        
        # Then fix any list-to-tensor issues
        for key, value in batch.items():
            if isinstance(value, list):
                # Convert list to tensor
                batch[key] = torch.tensor(value)
            elif isinstance(value, (tuple, list)) and len(value) > 0 and isinstance(value[0], list):
                # Handle nested lists
                batch[key] = torch.tensor(value)
        
        return batch


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main training function"""
    
    # Enable CUDA debugging for better error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    logger = get_logger()
    seed_everything(42)
    
    # Clear initial GPU memory
    clear_gpu_memory()
    
    # Configuration - keeping your original 32B model
    model_id_or_path = 'Qwen/Qwen2.5-VL-32B-Instruct'
    output_dir = 'output'
    data_seed = 42
    
    # Scale max_length based on GPU count for better memory distribution
    if gpu_count >= 8:
        max_length = 12288  # Your original values
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
    
    # Debug tokenizer information
    debug_tokenizer_and_vocab(tokenizer, model)
    
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
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
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
    
    # Safe filtering with proper token validation (NO TRUNCATION)
    logger.info("Filtering samples by encoding validity and length...")
    train_processed = safe_filter_dataset(train_processed, template, tokenizer, max_length)
    val_processed = safe_filter_dataset(val_processed, template, tokenizer, max_length)
    
    logger.info(f"Filtered dataset sizes - Train: {len(train_processed)}, Val: {len(val_processed)}")
    
    # Check if we have enough data after filtering
    if len(train_processed) == 0:
        logger.error("No training samples remain after filtering! There may be systematic tokenization issues.")
        return
    
    # Convert to LazyLLMDataset
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=data_seed)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=data_seed)
    
    logger.info(f"Final dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)} samples")
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Test first sample to make sure everything works
    logger.info("Testing first sample encoding...")
    try:
        test_sample = train_dataset[0]
        print(f"First sample test successful")
    except Exception as e:
        logger.error(f"First sample test failed: {e}")
        return
    
    # Create custom data collator to handle list-to-tensor conversion
    base_collator = template.data_collator
    fixed_collator = FixedDataCollator(base_collator)
    
    # Train
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=fixed_collator,  # Use our fixed collator
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        # Test first batch before full training
        logger.info("Testing first training batch...")
        train_dataloader = trainer.get_train_dataloader()
        first_batch = next(iter(train_dataloader))
        
        # Move to device and test forward pass
        first_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in first_batch.items()}
        
        with torch.no_grad():
            outputs = model(**first_batch)
            logger.info(f"First batch test successful. Loss: {outputs.loss}")
        
        # If first batch works, proceed with training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Clear memory on failure
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