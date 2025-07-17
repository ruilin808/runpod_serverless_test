#!/usr/bin/env python3
"""
Memory-Optimized Qwen2-VL Fine-tuning Script for Table HTML Conversion with Multi-GPU Support
Enhanced with inference evaluation during training
"""

import os
import torch
import gc
import json
from typing import Dict, Any, List
import numpy as np

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
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class InferenceEvaluationCallback(TrainerCallback):
    """Custom callback to run inference on validation samples during training"""
    
    def __init__(self, eval_dataset, template, model, processor, num_samples=5, output_dir="output", 
                 print_full_results=False, print_comparison=True):
        self.eval_dataset = eval_dataset
        self.template = template
        self.model = model
        self.processor = processor
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.inference_results = []
        self.print_full_results = print_full_results  # Print full HTML in terminal
        self.print_comparison = print_comparison      # Print side-by-side comparison
        
        # Create inference results directory
        self.inference_dir = os.path.join(output_dir, "inference_results")
        os.makedirs(self.inference_dir, exist_ok=True)
    
    def print_separator(self, title="", width=80):
        """Print a formatted separator line"""
        if title:
            separator = f"{'='*10} {title} {'='*(width-len(title)-12)}"
        else:
            separator = "="*width
        print(separator)
    
    def print_html_comparison(self, ground_truth, prediction, sample_idx):
        """Print a formatted comparison of ground truth vs prediction"""
        print(f"\n📊 SAMPLE {sample_idx} - HTML COMPARISON")
        print("─" * 80)
        
        # Basic stats
        gt_len = len(ground_truth)
        pred_len = len(prediction)
        print(f"📏 Length: GT={gt_len} chars | Pred={pred_len} chars | Diff={pred_len-gt_len:+d}")
        
        # HTML tag counts (basic analysis)
        gt_table_count = ground_truth.count('<table')
        pred_table_count = prediction.count('<table')
        gt_tr_count = ground_truth.count('<tr')
        pred_tr_count = prediction.count('<tr')
        gt_td_count = ground_truth.count('<td')
        pred_td_count = prediction.count('<td')
        
        print(f"🏷️  Tags: <table> GT={gt_table_count} Pred={pred_table_count} | <tr> GT={gt_tr_count} Pred={pred_tr_count} | <td> GT={gt_td_count} Pred={pred_td_count}")
        
        if self.print_comparison:
            print("\n🎯 GROUND TRUTH (first 300 chars):")
            print("┌" + "─" * 78 + "┐")
            gt_preview = ground_truth[:300] + "..." if len(ground_truth) > 300 else ground_truth
            for line in gt_preview.split('\n'):
                print(f"│ {line[:76]:<76} │")
            print("└" + "─" * 78 + "┘")
            
            print("\n🤖 PREDICTION (first 300 chars):")
            print("┌" + "─" * 78 + "┐")
            pred_preview = prediction[:300] + "..." if len(prediction) > 300 else prediction
            for line in pred_preview.split('\n'):
                print(f"│ {line[:76]:<76} │")
            print("└" + "─" * 78 + "┘")
        
        if self.print_full_results:
            print("\n📋 FULL GROUND TRUTH:")
            print("─" * 40)
            print(ground_truth)
            print("\n📋 FULL PREDICTION:")
            print("─" * 40)
            print(prediction)
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Run inference when evaluation is triggered"""
        if state.global_step == 0:
            return
            
        logger = get_logger()
        
        # Print evaluation header
        self.print_separator(f"INFERENCE EVALUATION - STEP {state.global_step}")
        print(f"🚀 Running inference on {self.num_samples} validation samples...")
        print(f"⏰ Step: {state.global_step} | Epoch: {state.epoch:.2f}")
        
        # Set model to eval mode
        self.model.eval()
        
        # Select random samples for inference
        sample_indices = np.random.choice(
            len(self.eval_dataset.dataset), 
            min(self.num_samples, len(self.eval_dataset.dataset)), 
            replace=False
        )
        
        inference_results = []
        
        with torch.no_grad():
            for i, idx in enumerate(sample_indices):
                try:
                    print(f"\n🔄 Processing sample {i+1}/{len(sample_indices)} (dataset idx: {idx})...")
                    
                    # Get the original sample
                    sample = self.eval_dataset.dataset[idx]
                    
                    # Prepare input for generation
                    messages = [
                        {
                            'role': 'user',
                            'content': 'Write the HTML representation for this image of a medical table.'
                        }
                    ]
                    
                    # Create generation input
                    generation_input = {
                        'messages': messages,
                        'images': sample['images']
                    }
                    
                    print("⚡ Generating prediction...")
                    
                    # Generate prediction
                    generated_ids = self.template.generate(
                        generation_input,
                        max_new_tokens=2048,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        num_beams=1,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    
                    # Decode prediction
                    prediction = self.processor.tokenizer.decode(
                        generated_ids[0], 
                        skip_special_tokens=True
                    )
                    
                    # Get ground truth
                    ground_truth = sample['messages'][1]['content']  # Assistant's response
                    
                    # Store results
                    result = {
                        'step': state.global_step,
                        'sample_idx': int(idx),
                        'ground_truth': ground_truth,
                        'prediction': prediction,
                        'ground_truth_length': len(ground_truth),
                        'prediction_length': len(prediction)
                    }
                    
                    inference_results.append(result)
                    
                    # Print detailed comparison
                    self.print_html_comparison(ground_truth, prediction, idx)
                    
                    print(f"✅ Sample {i+1} completed!")
                    
                except Exception as e:
                    print(f"❌ Error during inference on sample {idx}: {str(e)}")
                    logger.error(f"Error during inference on sample {idx}: {str(e)}")
                    continue
        
        # Print summary
        self.print_separator("INFERENCE SUMMARY")
        successful_samples = len(inference_results)
        avg_gt_length = sum(r['ground_truth_length'] for r in inference_results) / max(1, successful_samples)
        avg_pred_length = sum(r['prediction_length'] for r in inference_results) / max(1, successful_samples)
        
        print(f"📈 Results Summary:")
        print(f"   • Successful samples: {successful_samples}/{self.num_samples}")
        print(f"   • Average GT length: {avg_gt_length:.0f} chars")
        print(f"   • Average prediction length: {avg_pred_length:.0f} chars")
        print(f"   • Length difference: {avg_pred_length - avg_gt_length:+.0f} chars")
        
        # Save inference results
        results_file = os.path.join(self.inference_dir, f"inference_step_{state.global_step}.json")
        with open(results_file, 'w') as f:
            json.dump(inference_results, f, indent=2)
        
        # Store in callback for later analysis
        self.inference_results.extend(inference_results)
        
        # Set model back to train mode
        self.model.train()
        
        print(f"💾 Results saved to: {results_file}")
        self.print_separator("INFERENCE COMPLETE")
        print()  # Add spacing before training continues


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
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Create inference evaluation callback
    inference_callback = InferenceEvaluationCallback(
        eval_dataset=val_dataset,
        template=template,
        model=model,
        processor=processor,
        num_samples=5,  # Number of samples to run inference on
        output_dir=output_dir
    )
    
    # Train
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        callbacks=[inference_callback],  # Add the inference callback
    )
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        trainer.train()
        
        # Save final inference results summary
        final_results_file = os.path.join(output_dir, "final_inference_summary.json")
        with open(final_results_file, 'w') as f:
            json.dump(inference_callback.inference_results, f, indent=2)
        
        logger.info(f"Final inference results saved to {final_results_file}")
        
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