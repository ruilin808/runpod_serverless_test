#!/usr/bin/env python3
"""
Memory-Optimized Qwen2-VL Fine-tuning Script for Table HTML Conversion with Multi-GPU Support
WITH TEDS EVALUATION!!!!!!!!
"""

import os
import torch
import gc
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm

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

# TEDS imports
try:
    from table_recognition_metric import TEDS
    TEDS_AVAILABLE = True
except ImportError:
    print("WARNING: table_recognition_metric not installed. TEDS evaluation will be disabled.")
    print("Install with: pip install table-recognition-metric")
    TEDS_AVAILABLE = False


class TEDSMetric:
    """TEDS metric wrapper for evaluation"""
    
    def __init__(self, structure_only: bool = False, n_jobs: int = 1):
        """
        Initialize TEDS metric
        
        Args:
            structure_only: If True, only evaluate table structure (ignore content)
            n_jobs: Number of parallel jobs for evaluation
        """
        if not TEDS_AVAILABLE:
            raise ImportError("table_recognition_metric is not installed")
        
        self.teds = TEDS(structure_only=structure_only, n_jobs=n_jobs)
        self.structure_only = structure_only
        
    def compute_teds_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute TEDS score for predictions vs references
        
        Args:
            predictions: List of predicted HTML strings
            references: List of reference HTML strings
            
        Returns:
            Dictionary containing TEDS scores and statistics
        """
        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must match")
        
        scores = []
        valid_pairs = 0
        
        for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Computing TEDS"):
            try:
                # Clean HTML strings
                pred_clean = self._clean_html(pred)
                ref_clean = self._clean_html(ref)
                
                # Compute TEDS score
                score = self.teds.evaluate(pred_clean, ref_clean)
                scores.append(score)
                valid_pairs += 1
                
            except Exception as e:
                # Handle malformed HTML gracefully
                logging.warning(f"Error computing TEDS for pair: {e}")
                scores.append(0.0)  # Assign 0 score for failed cases
        
        # Calculate statistics
        scores = np.array(scores)
        results = {
            'teds_score': float(np.mean(scores)),
            'teds_std': float(np.std(scores)),
            'teds_min': float(np.min(scores)),
            'teds_max': float(np.max(scores)),
            'valid_pairs': valid_pairs,
            'total_pairs': len(predictions),
            'success_rate': valid_pairs / len(predictions) if len(predictions) > 0 else 0.0
        }
        
        return results
    
    def _clean_html(self, html_string: str) -> str:
        """Clean HTML string for TEDS evaluation"""
        if not html_string:
            return "<table></table>"
        
        # Remove common artifacts from model output
        html_string = html_string.strip()
        
        # Ensure the HTML is wrapped in a table tag if it's not already
        if not html_string.startswith('<table'):
            if '<table' in html_string:
                # Extract table content
                start_idx = html_string.find('<table')
                html_string = html_string[start_idx:]
            else:
                # Wrap in table tags if no table found
                html_string = f"<table>{html_string}</table>"
        
        # Handle truncated HTML (ending with "...")
        if html_string.endswith("..."):
            html_string = html_string[:-3]
            # Try to close any open tags
            if html_string.count('<tr') > html_string.count('</tr>'):
                html_string += '</tr>'
            if html_string.count('<td') > html_string.count('</td>'):
                html_string += '</td>'
            if not html_string.endswith('</table>'):
                html_string += '</table>'
        
        return html_string


class TEDSEvaluator:
    """Custom evaluator for TEDS during training"""
    
    def __init__(self, model, processor, template, teds_metric: TEDSMetric, 
                 max_new_tokens: int = 2048, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.template = template
        self.teds_metric = teds_metric
        self.max_new_tokens = max_new_tokens
        self.device = device
        
    def evaluate_batch(self, batch_data: List[Dict[str, Any]], batch_size: int = 4) -> Dict[str, float]:
        """
        Evaluate a batch of samples using TEDS
        
        Args:
            batch_data: List of samples with 'images' and 'html_table' keys
            batch_size: Batch size for inference
            
        Returns:
            Dictionary with TEDS evaluation results
        """
        predictions = []
        references = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i:i + batch_size]
            batch_preds = self._generate_batch_predictions(batch)
            predictions.extend(batch_preds)
            references.extend([sample['html_table'] for sample in batch])
        
        # Compute TEDS scores
        return self.teds_metric.compute_teds_score(predictions, references)
    
    def _generate_batch_predictions(self, batch: List[Dict[str, Any]]) -> List[str]:
        """Generate predictions for a batch of samples"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for sample in batch:
                try:
                    # Prepare input
                    messages = [{
                        'role': 'user',
                        'content': 'Write the HTML representation for this image of a medical table.'
                    }]
                    
                    # Create input with image
                    inputs = self.template.encode({
                        'messages': messages,
                        'images': sample['images']
                    })
                    
                    # Move to device
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    # Generate
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )
                    
                    # Decode prediction
                    response = self.processor.tokenizer.decode(
                        outputs[0][len(inputs['input_ids'][0]):],
                        skip_special_tokens=True
                    )
                    
                    predictions.append(response.strip())
                    
                except Exception as e:
                    logging.warning(f"Error generating prediction: {e}")
                    predictions.append("")
        
        return predictions


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


def run_teds_evaluation(model, processor, template, val_dataset_raw, output_dir, 
                       eval_samples=100, structure_only=False):
    """
    Run TEDS evaluation on validation set
    
    Args:
        model: The trained model
        processor: Model processor
        template: Template for encoding
        val_dataset_raw: Raw validation dataset
        output_dir: Output directory for results
        eval_samples: Number of samples to evaluate (None for all)
        structure_only: Whether to evaluate structure only
    """
    if not TEDS_AVAILABLE:
        logging.warning("TEDS evaluation skipped - table_recognition_metric not installed")
        return {}
    
    logging.info(f"Starting TEDS evaluation with {eval_samples or 'all'} samples...")
    
    # Initialize TEDS metric
    teds_metric = TEDSMetric(structure_only=structure_only, n_jobs=min(4, os.cpu_count()))
    
    # Initialize evaluator
    evaluator = TEDSEvaluator(model, processor, template, teds_metric)
    
    # Sample evaluation data
    eval_data = val_dataset_raw
    if eval_samples and len(eval_data) > eval_samples:
        eval_data = eval_data.shuffle(seed=42).select(range(eval_samples))
    
    # Convert to list of dictionaries
    eval_samples_list = []
    for sample in eval_data:
        eval_samples_list.append({
            'images': [sample['image']],
            'html_table': sample['html_table']
        })
    
    # Run evaluation
    results = evaluator.evaluate_batch(eval_samples_list, batch_size=2)
    
    # Save results
    results_path = os.path.join(output_dir, 'teds_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log results
    logging.info("TEDS Evaluation Results:")
    logging.info(f"  Mean TEDS Score: {results['teds_score']:.4f}")
    logging.info(f"  Standard Deviation: {results['teds_std']:.4f}")
    logging.info(f"  Min Score: {results['teds_min']:.4f}")
    logging.info(f"  Max Score: {results['teds_max']:.4f}")
    logging.info(f"  Success Rate: {results['success_rate']:.4f}")
    logging.info(f"  Valid Pairs: {results['valid_pairs']}/{results['total_pairs']}")
    
    return results


class CustomTrainer(Seq2SeqTrainer):
    """Custom trainer with TEDS evaluation support"""
    
    def __init__(self, teds_evaluator=None, eval_steps_teds=500, **kwargs):
        super().__init__(**kwargs)
        self.teds_evaluator = teds_evaluator
        self.eval_steps_teds = eval_steps_teds
        self.teds_eval_count = 0
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to include TEDS evaluation"""
        # Run standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Run TEDS evaluation periodically
        if (self.teds_evaluator is not None and 
            TEDS_AVAILABLE and 
            self.state.global_step % self.eval_steps_teds == 0):
            
            try:
                # Sample a small subset for TEDS evaluation during training
                eval_samples = 20  # Small subset to avoid slowing down training
                teds_results = self.run_teds_evaluation_subset(eval_samples)
                
                # Add TEDS scores to evaluation results
                eval_results.update({
                    f"{metric_key_prefix}_teds_score": teds_results.get('teds_score', 0.0),
                    f"{metric_key_prefix}_teds_success_rate": teds_results.get('success_rate', 0.0)
                })
                
                self.teds_eval_count += 1
                
            except Exception as e:
                logging.warning(f"TEDS evaluation failed: {e}")
        
        return eval_results
    
    def run_teds_evaluation_subset(self, num_samples):
        """Run TEDS evaluation on a subset of validation data"""
        # This would need access to the raw validation dataset
        # For now, return dummy results
        return {'teds_score': 0.0, 'success_rate': 0.0}


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
    
    # TEDS evaluation settings
    enable_teds_eval = TEDS_AVAILABLE
    teds_eval_samples = 100  # Number of samples for final TEDS evaluation
    teds_structure_only = False  # Set to True to evaluate only table structure
    
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
    logger.info(f"  - TEDS evaluation enabled: {enable_teds_eval}")
    
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
        save_steps=200,  # Increased to reduce I/O
        eval_strategy='steps',
        eval_steps=200,  # Increased to reduce I/O
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
        # Evaluation settings
        load_best_model_at_end=True,
        evaluation_strategy='steps',
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
    
    # Keep raw validation dataset for TEDS evaluation
    val_dataset_raw = raw_dataset['validation']
    
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
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Initialize TEDS evaluator for periodic evaluation (if enabled)
    teds_evaluator = None
    if enable_teds_eval:
        try:
            teds_metric = TEDSMetric(structure_only=teds_structure_only, n_jobs=2)
            teds_evaluator = TEDSEvaluator(model, processor, template, teds_metric)
            logger.info("TEDS evaluator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize TEDS evaluator: {e}")
            teds_evaluator = None
    
    # Train
    model.enable_input_require_grads()
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        teds_evaluator=teds_evaluator,
        eval_steps_teds=500,  # Run TEDS evaluation every 500 steps
    )
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        trainer.train()
        
        # Run final comprehensive TEDS evaluation
        if enable_teds_eval and teds_evaluator is not None:
            logger.info("Running final TEDS evaluation...")
            final_teds_results = run_teds_evaluation(
                model=model,
                processor=processor,
                template=template,
                val_dataset_raw=val_dataset_raw,
                output_dir=output_dir,
                eval_samples=teds_eval_samples,
                structure_only=teds_structure_only
            )
            
            # Save final results summary
            final_results = {
                'training_completed': True,
                'final_model_checkpoint': trainer.state.best_model_checkpoint,
                'teds_evaluation': final_teds_results,
                'training_args': training_args.to_dict(),
                'model_parameters': model_parameter_info
            }
            
            results_path = os.path.join(output_dir, 'final_results.json')
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info(f"Final TEDS Score: {final_teds_results.get('teds_score', 0.0):.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Clear memory on failure
        clear_gpu_memory()
        return
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Model saved to: {trainer.state.best_model_checkpoint}')
    
    # Final memory cleanup
    clear_gpu_memory()


if __name__ == "__main__":
    main()