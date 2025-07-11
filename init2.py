#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script for Table HTML Conversion with TEDS Evaluation
"""

import os
import psutil
import torch
import time
import numpy as np
from typing import Dict, List, Optional

# Use all available GPUs
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
    print(f"Using {gpu_count} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("CUDA not available, using CPU")

from swift.llm import (
    get_model_tokenizer, get_template,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset as hf_load_dataset
from transformers import TrainerCallback

# TEDS import
try:
    from table_recognition_metric import TEDS
    print("TEDS library loaded successfully")
except ImportError:
    print("TEDS library not found. Install with: pip install table-recognition-metric")
    raise

# Constants
MAX_EVAL_SAMPLES_TRAINING = 3
MAX_EVAL_SAMPLES_FINAL = 10
RESOURCE_LOG_INTERVAL = 10
EVAL_GENERATION_MAX_TOKENS = 8192


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


def get_device():
    """Get the appropriate device for inference"""
    if torch.cuda.is_available():
        return f'cuda:{torch.cuda.current_device()}'
    return 'cpu'


def log_resources(logger, step="", samples_per_sec=None):
    """Log system resources and training metrics"""
    # GPU information
    if torch.cuda.is_available():
        gpu_info_parts = []
        for i in range(torch.cuda.device_count()):
            gpu_mem_used = torch.cuda.memory_allocated(i) / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_info_parts.append(f"GPU{i}: {gpu_mem_used:.1f}/{gpu_mem_total:.1f}GB")
        gpu_info = " | ".join(gpu_info_parts)
    else:
        gpu_info = "N/A"
    
    # System resources
    ram = psutil.virtual_memory()
    ram_used = ram.used / 1024**3
    ram_total = ram.total / 1024**3
    ram_percent = ram.percent
    
    cpu_percent = psutil.cpu_percent()
    cpu_count = psutil.cpu_count()
    
    disk = psutil.disk_usage('/')
    disk_used = disk.used / 1024**3
    disk_total = disk.total / 1024**3
    disk_percent = (disk_used / disk_total) * 100
    
    # Format training speed
    speed_info = f" | Speed: {samples_per_sec:.3f} samples/s" if samples_per_sec else ""
    
    logger.info(f"[{step}] {gpu_info} | RAM: {ram_used:.1f}/{ram_total:.1f}GB ({ram_percent:.1f}%) | "
                f"CPU: {cpu_percent:.1f}% ({cpu_count} cores) | Disk: {disk_used:.1f}/{disk_total:.1f}GB ({disk_percent:.1f}%){speed_info}")


def handle_generation_error(logger, error, context=""):
    """Centralized error handling for generation"""
    logger.warning(f"Generation error in {context}: {error}")
    return ""


class TEDSEvaluator:
    """TEDS evaluator for table structure recognition"""
    
    def __init__(self, model, processor, template, logger, device=None):
        self.model = model
        self.processor = processor
        self.template = template
        self.logger = logger
        self.device = device or get_device()
        
        # Initialize TEDS scorers
        self.teds_structure_content = TEDS(structure_only=False)  # Structure + Content
        self.teds_structure_only = TEDS(structure_only=True)      # Structure only
        
        self.logger.info("TEDS evaluators initialized")
        self.logger.info("  - Structure + Content: TEDS(structure_only=False)")
        self.logger.info("  - Structure only: TEDS(structure_only=True)")
    
    def generate_prediction(self, sample, max_new_tokens=EVAL_GENERATION_MAX_TOKENS):
        """Generate HTML prediction for a sample"""
        try:
            self.model.eval()
            
            with torch.no_grad():
                # Prepare input sample
                input_sample = {
                    'messages': [sample['messages'][0]],  # Only user message
                    'images': sample.get('images', [])
                }
                
                # Encode with template
                inputs = self.template.encode(input_sample)
                
                # Convert to tensors
                input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long, device=self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                # Prepare generation kwargs
                generation_kwargs = {
                    'device': self.device,
                    'use_cache': False,
                    'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
                }
                
                # Add other inputs if present
                for key, tensor_key in [('attention_mask', 'attention_mask'), 
                                      ('pixel_values', 'pixel_values'),
                                      ('image_grid_thw', 'image_grid_thw')]:
                    if key in inputs:
                        tensor = torch.tensor(inputs[key], device=self.device)
                        if tensor.dim() == 1 and key != 'pixel_values':
                            tensor = tensor.unsqueeze(0)
                        generation_kwargs[tensor_key] = tensor
                
                # Generate prediction
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    **generation_kwargs
                )
                
                # Decode prediction
                input_length = input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
                prediction = self.processor.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                return prediction.strip()
                
        except Exception as e:
            return handle_generation_error(self.logger, e, "TEDSEvaluator.generate_prediction")
    
    def evaluate_samples(self, eval_dataset, max_samples):
        """Evaluate TEDS scores on a subset of samples"""
        self.logger.info(f"TEDS EVALUATION ON {min(max_samples, len(eval_dataset))} SAMPLES")
        
        structure_content_scores = []
        structure_only_scores = []
        successful_evaluations = 0
        
        for i in range(min(max_samples, len(eval_dataset))):
            try:
                sample = eval_dataset[i]
                
                # Generate prediction
                self.logger.info(f"Sample {i+1}: Generating prediction...")
                pred_html = self.generate_prediction(sample)
                
                # Get ground truth
                if 'messages' in sample and len(sample['messages']) > 1:
                    true_html = sample['messages'][1]['content']
                elif 'html_table' in sample:
                    true_html = sample['html_table']
                else:
                    self.logger.warning(f"Sample {i+1}: No ground truth found")
                    continue
                
                # Validate HTML strings
                if not pred_html.strip() or not true_html.strip():
                    self.logger.warning(f"Sample {i+1}: Empty HTML (pred: {len(pred_html)}, true: {len(true_html)})")
                    continue
                
                # Calculate TEDS scores
                try:
                    teds_sc_score = self.teds_structure_content(true_html, pred_html)
                    structure_content_scores.append(teds_sc_score)
                    
                    teds_so_score = self.teds_structure_only(true_html, pred_html)
                    structure_only_scores.append(teds_so_score)
                    
                    successful_evaluations += 1
                    
                    self.logger.info(f"Sample {i+1} TEDS Scores - Structure+Content: {teds_sc_score:.4f}, Structure: {teds_so_score:.4f}")
                    
                except Exception as score_error:
                    self.logger.warning(f"Sample {i+1}: TEDS scoring error: {score_error}")
                    continue
                    
            except Exception as e:
                self.logger.warning(f"Sample {i+1}: Error: {e}")
                continue
        
        # Calculate averages
        avg_structure_content = np.mean(structure_content_scores) if structure_content_scores else 0.0
        avg_structure_only = np.mean(structure_only_scores) if structure_only_scores else 0.0
        
        self.logger.info(f"TEDS EVALUATION RESULTS:")
        self.logger.info(f"Successfully evaluated: {successful_evaluations}/{max_samples} samples")
        self.logger.info(f"Average TEDS (Structure + Content): {avg_structure_content:.4f}")
        self.logger.info(f"Average TEDS (Structure only): {avg_structure_only:.4f}")
        
        return {
            'eval_teds_structure_content': avg_structure_content,
            'eval_teds_structure_only': avg_structure_only,
            'eval_teds_samples': successful_evaluations
        }


class ResourceCallback(TrainerCallback):
    """Enhanced callback for resource logging and TEDS evaluation"""
    def __init__(self, logger, teds_evaluator=None, eval_dataset=None):
        self.logger = logger
        self.teds_evaluator = teds_evaluator
        self.eval_dataset = eval_dataset
        self.last_time = None
        self.last_step = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % RESOURCE_LOG_INTERVAL == 0:
            # Calculate samples per second
            if self.last_time and state.global_step > self.last_step:
                current_time = time.time()
                steps_elapsed = state.global_step - self.last_step
                time_elapsed = current_time - self.last_time
                samples_per_sec = (steps_elapsed * args.per_device_train_batch_size * args.gradient_accumulation_steps) / time_elapsed
                log_resources(self.logger, f"Step-{state.global_step}", samples_per_sec)
                self.last_time = current_time
                self.last_step = state.global_step
            else:
                self.last_time = time.time()
                self.last_step = state.global_step
                log_resources(self.logger, f"Step-{state.global_step}")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Run TEDS evaluation during training evaluation"""
        if self.teds_evaluator and self.eval_dataset and state.global_step > 0:
            try:
                teds_results = self.teds_evaluator.evaluate_samples(self.eval_dataset, MAX_EVAL_SAMPLES_TRAINING)
                
                # Add to logs
                if logs is not None:
                    logs.update(teds_results)
                
                self.logger.info(f"[Step {state.global_step}] TEDS Evaluation:")
                self.logger.info(f"  Structure + Content: {teds_results['eval_teds_structure_content']:.4f}")
                self.logger.info(f"  Structure only: {teds_results['eval_teds_structure_only']:.4f}")
                self.logger.info(f"  Samples evaluated: {teds_results['eval_teds_samples']}")
                
            except Exception as e:
                self.logger.error(f"TEDS evaluation failed: {e}")


def main():
    """Main training function"""
    
    logger = get_logger()
    seed_everything(42)
    
    # Configuration
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
    output_dir = 'output'
    data_seed = 42
    max_length = 32768
    
    # LoRA configuration
    lora_rank = 4
    lora_alpha = 16
    freeze_llm = False
    freeze_vit = True
    freeze_aligner = True
    
    # Calculate optimal batch sizes for multi-GPU
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    per_device_batch_size = 1
    gradient_accumulation_steps = max(8 // gpu_count, 1)  # Maintain effective batch size
    
    logger.info(f"Multi-GPU setup: {gpu_count} GPUs, batch_size={per_device_batch_size}, grad_accum={gradient_accumulation_steps}")
    
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
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=50,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        metric_for_best_model='eval_teds_structure_content',
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
        data_seed=data_seed,
        remove_unused_columns=False,
        bf16=True,
        optim='adafactor',
        max_grad_norm=1.0,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,  # Optimize for multi-GPU
    )
    
    # Log initial resources
    log_resources(logger, "START")
    
    # Load model and setup template
    logger.info("Loading model and tokenizer...")
    model, processor = get_model_tokenizer(model_id_or_path)
    template = get_template(model.model_meta.template, processor, default_system=None, max_length=max_length)
    template.set_mode('train')
    if template.use_model:
        template.model = model
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                freeze_aligner=freeze_aligner)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'Model parameter info: {model_parameter_info}')
    
    # Load and process dataset
    logger.info("Loading and processing dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    train_processed = raw_dataset['train'].map(create_swift_format_single)
    val_processed = raw_dataset['validation'].map(create_swift_format_single)
    
    # Convert to LazyLLMDataset
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=data_seed)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=data_seed)
    
    logger.info(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)} samples")
    
    # Initialize TEDS evaluator
    logger.info("Initializing TEDS evaluator...")
    device = get_device()
    teds_evaluator = TEDSEvaluator(model, processor, template, logger, device)
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train
    logger.info("Starting training...")
    model.enable_input_require_grads()
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        callbacks=[ResourceCallback(logger, teds_evaluator, val_processed)],
    )
    
    trainer.train()
    
    # Final TEDS evaluation
    logger.info("Running final TEDS evaluation...")
    final_teds_results = teds_evaluator.evaluate_samples(val_processed, MAX_EVAL_SAMPLES_FINAL)
    
    # Log final resources
    log_resources(logger, "END")
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Model saved to: {trainer.state.last_model_checkpoint}')
    logger.info(f'Final TEDS (Structure + Content): {final_teds_results["eval_teds_structure_content"]:.4f}')
    logger.info(f'Final TEDS (Structure only): {final_teds_results["eval_teds_structure_only"]:.4f}')


if __name__ == "__main__":
    main()