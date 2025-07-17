#!/usr/bin/env python3
"""
Memory-Optimized Qwen2-VL Fine-tuning Script for Table HTML Conversion with Multi-GPU Support
NO TEDS WORKING!!!!!!!!
"""

import os
import torch
import gc
import numpy as np
from transformers import TrainerCallback

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


class InferenceCallback(TrainerCallback):
    """Callback to run inference on validation samples during evaluation"""
    
    def __init__(self, eval_dataset, template, processor, num_samples=3, max_gen_length=512):
        self.eval_dataset = eval_dataset
        self.template = template
        self.processor = processor  # Store processor/tokenizer
        self.num_samples = min(num_samples, len(eval_dataset)) if len(eval_dataset) > 0 else 0
        self.max_gen_length = max_gen_length
        
        # Pre-select samples to maintain consistency across evaluations
        if self.num_samples > 0:
            np.random.seed(42)  # Fixed seed for reproducible samples
            self.sample_indices = np.random.choice(
                len(self.eval_dataset), 
                self.num_samples, 
                replace=False
            )
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Called at the end of evaluation - runs inference on validation samples"""
        
        # Only run on main process to avoid duplicated output
        if args.local_rank not in [-1, 0]:
            return
            
        # Skip if no samples to evaluate
        if self.num_samples == 0:
            return
            
        try:
            print(f"\n🔍 Running inference validation at step {state.global_step}")
            print("=" * 70)
            
            # Ensure model is in eval mode
            model.eval()
            
            with torch.no_grad():
                for i, idx in enumerate(self.sample_indices):
                    try:
                        sample = self.eval_dataset[idx]
                        
                        # Prepare input (just the user message for inference)
                        user_message = sample['messages'][0]['content']
                        inference_sample = {
                            'messages': [{'role': 'user', 'content': user_message}],
                            'images': sample.get('images', [])
                        }
                        
                        # Encode input
                        inputs = self.template.encode(inference_sample)
                        
                        # Move to device and add batch dimension
                        input_ids = inputs['input_ids'].unsqueeze(0).to(model.device)
                        attention_mask = inputs['attention_mask'].unsqueeze(0).to(model.device)
                        
                        # Generate with conservative settings
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=min(self.max_gen_length, len(inputs['input_ids']) + 256),
                            do_sample=False,  # Deterministic
                            num_beams=1,      # Faster than beam search
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )
                        
                        # Decode prediction (remove input tokens)
                        generated_tokens = outputs[0][len(input_ids[0]):]
                        predicted_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                        # Get ground truth
                        ground_truth = sample['messages'][1]['content']
                        
                        # Display results
                        print(f"\n--- Validation Sample {i+1} (Index: {idx}) ---")
                        print(f"Input: {user_message}")
                        print(f"Prediction: {predicted_text[:300]}{'...' if len(predicted_text) > 300 else ''}")
                        print(f"Ground Truth: {ground_truth[:300]}{'...' if len(ground_truth) > 300 else ''}")
                        
                        # Simple quality check
                        if predicted_text.strip():
                            print(f"✅ Generated non-empty response")
                        else:
                            print(f"❌ Generated empty response")
                            
                    except Exception as e:
                        print(f"❌ Failed to process sample {idx}: {e}")
                        continue
                        
            print("=" * 70)
            print(f"✅ Inference validation completed at step {state.global_step}\n")
            
        except Exception as e:
            print(f"❌ Inference callback failed: {e}")
            # Training continues normally
        
        finally:
            # Ensure we're back in training mode
            model.train()
            # Clear any accumulated memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
        save_steps=100,  # Increased to reduce I/O
        eval_strategy='steps',
        eval_steps=2,  # Increased to reduce I/O
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        metric_for_best_model='loss',
        save_total_limit=3,  # Reduced to save disk space
        logging_steps=10,  # Increased to reduce overhead
        dataloader_num_workers=dataloader_workers,  # Scale with GPU count
        data_seed=data_seed,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        # Memory optimization settings
        dataloader_pin_memory=False,  # Disable to save memory
        ddp_find_unused_parameters=False,
        ddp_timeout=1800,
        # Additional memory optimizations
        fp16=True,  # Enable mixed precision training
        dataloader_prefetch_factor=2,  # Reduce prefetch to save memory
        ddp_bucket_cap_mb=25,  # Reduce DDP bucket size
        save_safetensors=True,  # More efficient saving
    )
    
    # Load model with memory optimizations
    logger.info("Loading model with memory optimizations...")
    model, processor = get_model_tokenizer(
        model_id_or_path,
        torch_dtype=torch.float16,  # Use half precision
        device_map='auto',  # Let accelerate handle device placement
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
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
        lora_dropout=0.1,  # Add dropout for regularization
        bias="none",  # Don't adapt bias terms to save memory
    )
    model = Swift.prepare_model(model, lora_config)
    
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')
    
    # Clear memory after model loading
    clear_gpu_memory()
    
    # Load and process dataset
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Apply HTML truncation to reduce sequence length
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
        logger.error("No training samples remain after filtering! Consider increasing max_length or reducing HTML truncation.")
        return
    
    # Create inference callback
    # Scale number of inference samples based on validation dataset size
    inference_samples = min(3, len(val_dataset))  # Don't exceed available samples
    inference_callback = InferenceCallback(
        eval_dataset=val_processed,  # Use original processed dataset, not LazyLLMDataset
        template=template,
        processor=processor,  # Pass processor instead of relying on tokenizer parameter
        num_samples=inference_samples,
        max_gen_length=512  # Conservative generation length
    )
    
    logger.info(f"Inference callback configured with {inference_samples} validation samples")
    
    # Clear memory before training
    clear_gpu_memory()
    
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
    
    # Add the inference callback
    trainer.add_callback(inference_callback)
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        logger.info("Starting training with inference validation callback...")
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