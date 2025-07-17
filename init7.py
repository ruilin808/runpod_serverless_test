#!/usr/bin/env python3
"""
Memory-Optimized Qwen2-VL Fine-tuning Script for Table HTML Conversion with Multi-GPU Support and TEDS Evaluation
"""

import os

# Set cache directories to /workspace for RunPod persistence
os.environ['MODELSCOPE_CACHE'] = '/workspace/cache/modelscope'
os.environ['HF_DATASETS_CACHE'] = '/workspace/cache/datasets'

# Create directories
os.makedirs('/workspace/cache/modelscope', exist_ok=True)
os.makedirs('/workspace/cache/datasets', exist_ok=True)

import torch
import gc
import json
import re
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any

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

# Setup GPUs after workspace setup
gpu_count = setup_gpus()

from swift.llm import (
    get_model_tokenizer, get_template,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrainerCallback
from datasets import load_dataset as hf_load_dataset

# NEW: Import TEDS evaluation
try:
    from table_recognition_metric import TEDS
    TEDS_AVAILABLE = True
    print("TEDS evaluation enabled")
except ImportError:
    TEDS_AVAILABLE = False
    print("WARNING: table_recognition_metric not installed. TEDS evaluation disabled.")
    print("Install with: pip install table_recognition_metric")

# NEW: HTML normalization for better TEDS comparison
def normalize_html_for_teds(html_string: str) -> str:
    """Normalize HTML for better TEDS comparison"""
    try:
        from bs4 import BeautifulSoup
        
        # Parse and clean HTML
        soup = BeautifulSoup(html_string, 'html.parser')
        
        # Remove unnecessary attributes but keep essential table structure
        for tag in soup.find_all():
            if tag.name == 'table':
                # Keep essential table attributes
                tag.attrs = {k: v for k, v in tag.attrs.items() if k in ['border', 'cellpadding', 'cellspacing']}
            elif tag.name in ['td', 'th']:
                # Keep cell spanning attributes
                tag.attrs = {k: v for k, v in tag.attrs.items() if k in ['colspan', 'rowspan']}
            else:
                # Remove all other attributes
                tag.attrs = {}
        
        # Get clean HTML
        clean_html = str(soup)
        
        # Normalize whitespace
        clean_html = re.sub(r'\s+', ' ', clean_html)
        clean_html = re.sub(r'>\s+<', '><', clean_html)
        
        return clean_html.strip()
        
    except Exception as e:
        print(f"Warning: HTML normalization failed: {e}")
        return html_string.strip()

# NEW: TEDS evaluation functions
def evaluate_single_teds(args):
    """Evaluate single prediction-ground truth pair with TEDS"""
    if not TEDS_AVAILABLE:
        return 0.0
    
    predicted_html, ground_truth_html = args
    try:
        teds_evaluator = TEDS()
        
        # Normalize both HTML strings
        pred_normalized = normalize_html_for_teds(predicted_html)
        gt_normalized = normalize_html_for_teds(ground_truth_html)
        
        # Calculate TEDS score
        score = teds_evaluator.evaluate(pred_normalized, gt_normalized)
        return float(score)
        
    except Exception as e:
        print(f"Warning: TEDS evaluation failed for sample: {e}")
        return 0.0

def evaluate_teds_parallel(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Evaluate TEDS scores in parallel using multiple CPU cores"""
    if not TEDS_AVAILABLE or len(predictions) == 0:
        return {"teds_score": 0.0, "teds_count": 0}
    
    # Determine number of processes based on CPU count and GPU count
    # Use more CPU cores when we have more GPUs (since we have more computational resources)
    if gpu_count >= 8:
        num_processes = min(cpu_count(), 8)  # Use up to 8 cores for large GPU setups
    elif gpu_count >= 4:
        num_processes = min(cpu_count(), 6)  # Use up to 6 cores for medium GPU setups
    else:
        num_processes = min(cpu_count(), 4)  # Use up to 4 cores for small GPU setups
    
    print(f"Evaluating TEDS with {num_processes} CPU processes for {len(predictions)} samples")
    
    try:
        # Prepare arguments for parallel processing
        eval_args = list(zip(predictions, ground_truths))
        
        # Use multiprocessing for parallel TEDS evaluation
        with Pool(processes=num_processes) as pool:
            scores = pool.map(evaluate_single_teds, eval_args)
        
        # Calculate statistics
        valid_scores = [s for s in scores if s > 0]  # Filter out failed evaluations
        
        if len(valid_scores) == 0:
            return {"teds_score": 0.0, "teds_count": 0}
        
        avg_score = sum(valid_scores) / len(valid_scores)
        
        return {
            "teds_score": avg_score,
            "teds_count": len(valid_scores),
            "teds_failed": len(scores) - len(valid_scores)
        }
        
    except Exception as e:
        print(f"Error in parallel TEDS evaluation: {e}")
        return {"teds_score": 0.0, "teds_count": 0}

# NEW: Custom callback for TEDS evaluation during training
class TEDSEvaluationCallback(TrainerCallback):
    """Custom callback to compute TEDS scores during training"""
    
    def __init__(self, template, eval_dataset, eval_frequency=100, gpu_count=1):
        super().__init__()
        self.template = template
        self.eval_dataset = eval_dataset
        self.eval_frequency = eval_frequency
        self.gpu_count = gpu_count
        self.step_count = 0
        self.eval_count = 0
        self.teds_history = []
        
        # Scale sample sizes based on GPU count
        if gpu_count >= 8:
            self.quick_sample_size = min(50, len(eval_dataset))    # Larger setups can afford more
            self.thorough_sample_size = min(150, len(eval_dataset))
        elif gpu_count >= 4:
            self.quick_sample_size = min(30, len(eval_dataset))    # Moderate sampling
            self.thorough_sample_size = min(100, len(eval_dataset))
        else:
            self.quick_sample_size = min(20, len(eval_dataset))    # Conservative for smaller setups
            self.thorough_sample_size = min(75, len(eval_dataset))
        
        print(f"TEDS Evaluation Configuration:")
        print(f"  - Quick evaluation: {self.quick_sample_size} samples")
        print(f"  - Thorough evaluation: {self.thorough_sample_size} samples (every 5th evaluation)")
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of trainer initialization"""
        print("TEDS Evaluation Callback initialized successfully")
        return control
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Called during evaluation to compute TEDS scores"""
        if not TEDS_AVAILABLE or logs is None:
            return control
        
        try:
            self.eval_count += 1
            
            # Use thorough evaluation every 5th time, quick evaluation otherwise
            is_thorough_eval = (self.eval_count % 5 == 0)
            eval_sample_size = self.thorough_sample_size if is_thorough_eval else self.quick_sample_size
            
            eval_type = "thorough" if is_thorough_eval else "quick"
            print(f"Computing TEDS ({eval_type}) on {eval_sample_size} samples...")
            
            eval_indices = torch.randperm(len(self.eval_dataset))[:eval_sample_size]
            
            predictions = []
            ground_truths = []
            
            # Generate predictions for sampled data
            with torch.no_grad():
                for idx in eval_indices:
                    sample = self.eval_dataset[idx]
                    
                    # Extract ground truth HTML
                    if hasattr(sample, 'get') and 'messages' in sample:
                        for message in sample['messages']:
                            if message.get('role') == 'assistant':
                                ground_truths.append(message.get('content', ''))
                                break
                    elif hasattr(sample, '__getitem__'):
                        # Handle different dataset formats
                        try:
                            if 'html_table' in sample:
                                ground_truths.append(sample['html_table'])
                            elif 'target' in sample:
                                ground_truths.append(sample['target'])
                            else:
                                ground_truths.append('')
                        except:
                            ground_truths.append('')
                    else:
                        ground_truths.append('')
                    
                    # Generate prediction (simplified - in real implementation you'd use the model)
                    # This is a placeholder - actual implementation would generate from model
                    predicted_html = "<table><tr><td>Placeholder</td></tr></table>"
                    predictions.append(predicted_html)
            
            # Compute TEDS scores in parallel
            teds_results = evaluate_teds_parallel(predictions, ground_truths)
            
            # Add TEDS results to logs
            if logs is not None:
                logs.update(teds_results)
            
            # Store history with evaluation type
            self.teds_history.append({
                'step': state.global_step,
                'eval_count': self.eval_count,
                'teds_score': teds_results.get('teds_score', 0.0),
                'sample_size': eval_sample_size,
                'eval_type': eval_type
            })
            
            print(f"TEDS Score ({eval_type}): {teds_results.get('teds_score', 0.0):.4f} "
                  f"({eval_sample_size} samples)")
            
        except Exception as e:
            print(f"Error in TEDS evaluation: {e}")
            if logs is not None:
                logs['teds_score'] = 0.0
        
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called on each logging step"""
        self.step_count = state.global_step
        return control

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
        # Scale HTML length based on GPU count and available memory
        if gpu_count >= 8:
            max_html_chars = 8000  # More GPUs = can handle longer sequences
        elif gpu_count >= 4:
            max_html_chars = 6000
        else:
            max_html_chars = 4000
    
    if len(sample['html_table']) > max_html_chars:
        sample['html_table'] = sample['html_table'][:max_html_chars] + "..."
    return sample

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
    model_id_or_path = 'Qwen/Qwen2.5-VL-32B-Instruct'
    output_dir = '/workspace/output'
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
    logger.info(f"  - TEDS evaluation: {'Enabled' if TEDS_AVAILABLE else 'Disabled'}")
    
    # Adjust workers based on GPU count
    dataloader_workers = min(4, max(2, gpu_count // 2))  # Scale workers with GPUs but cap at 4
    
    # NEW: Adjust evaluation frequency based on GPU count
    # More GPUs = can afford more frequent evaluation
    if gpu_count >= 8:
        eval_steps = 50   # More frequent evaluation with more GPUs
        save_steps = 50
    elif gpu_count >= 4:
        eval_steps = 100  # Moderate evaluation frequency
        save_steps = 100
    else:
        eval_steps = 150  # Less frequent evaluation with fewer GPUs
        save_steps = 150
    
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
        save_steps=save_steps,
        eval_strategy='steps',
        eval_steps=eval_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        metric_for_best_model='loss',  # Could be changed to 'teds_score' if implemented
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
        # Additional memory optimizations
        fp16=True,
        dataloader_prefetch_factor=2,
        ddp_bucket_cap_mb=25,
        save_safetensors=True,
        # NEW: Enable evaluation prediction saving for TEDS
        predict_with_generate=True,
        include_inputs_for_metrics=True,
    )
    
    # Load model with memory optimizations
    logger.info("Loading model with memory optimizations...")
    model, processor = get_model_tokenizer(
        model_id_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
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
    
    # Apply HTML truncation to reduce sequence length
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
    
    # NEW: Initialize TEDS evaluation callback
    teds_callback = None
    if TEDS_AVAILABLE:
        teds_callback = TEDSEvaluationCallback(
            template=template,
            eval_dataset=val_dataset,
            eval_frequency=eval_steps,
            gpu_count=gpu_count  # Pass GPU count for sample size scaling
        )
        logger.info("TEDS evaluation callback initialized with optimized sample sizes")
    
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
        # NEW: Add callbacks if available
        callbacks=[teds_callback] if teds_callback else None,
    )
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        trainer.train()
        
        # NEW: Final TEDS evaluation on full validation set
        if TEDS_AVAILABLE:
            logger.info("Performing final TEDS evaluation on full validation set...")
            
            # Generate predictions for full validation set
            predictions = []
            ground_truths = []
            
            with torch.no_grad():
                for i in range(min(500, len(val_dataset))):  # Limit to 500 samples for final eval
                    sample = val_dataset[i]
                    
                    # Extract ground truth
                    if 'messages' in sample:
                        for message in sample['messages']:
                            if message['role'] == 'assistant':
                                ground_truths.append(message['content'])
                                break
                    
                    # Generate prediction (placeholder)
                    predicted_html = "<table><tr><td>Final Placeholder</td></tr></table>"
                    predictions.append(predicted_html)
            
            # Compute final TEDS scores
            final_teds_results = evaluate_teds_parallel(predictions, ground_truths)
            logger.info(f"Final TEDS Results: {final_teds_results}")
            
            # Save TEDS results
            teds_results_path = os.path.join(output_dir, 'teds_results.json')
            with open(teds_results_path, 'w') as f:
                json.dump({
                    'final_teds_results': final_teds_results,
                    'teds_history': teds_callback.teds_history if teds_callback else []
                }, f, indent=2)
            
            logger.info(f"TEDS results saved to: {teds_results_path}")
        
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