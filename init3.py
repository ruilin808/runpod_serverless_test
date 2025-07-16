#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script for Table HTML Conversion with TEDS Evaluation
Fixed version with CUDA error handling and validation
"""

import os
import psutil
import torch
import time
import numpy as np

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Use all available GPUs
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
    print(f"Using {gpu_count} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

from swift.llm import get_model_tokenizer, get_template, get_multimodal_target_regex, LazyLLMDataset
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


def create_swift_format_single(sample):
    """Convert single sample to Swift format with safety checks"""
    try:
        html_content = sample.get('html_table', '').strip()
        
        # Validate HTML content
        if not html_content or len(html_content) < 10:
            html_content = "<table></table>"  # Fallback for empty tables
            
        # Ensure content is reasonable length to prevent tokenization issues
        if len(html_content) > 30000:  # Adjust based on your needs
            html_content = html_content[:30000] + "..."
            
        # Additional safety check for image
        if 'image' not in sample or sample['image'] is None:
            return None
            
        return {
            'messages': [
                {'role': 'user', 'content': 'Write the HTML representation for this image of a medical table.'},
                {'role': 'assistant', 'content': html_content}
            ],
            'images': [sample['image']]
        }
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None


def log_resources(logger, step="", samples_per_sec=None):
    """Log system resources and training metrics"""
    # GPU info
    if torch.cuda.is_available():
        gpu_info = " | ".join([f"GPU{i}: {torch.cuda.memory_allocated(i)/1024**3:.1f}/{torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}GB" 
                              for i in range(torch.cuda.device_count())])
    else:
        gpu_info = "N/A"
    
    # System resources
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    speed_info = f" | Speed: {samples_per_sec:.3f} samples/s" if samples_per_sec else ""
    
    logger.info(f"[{step}] {gpu_info} | RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f}GB ({ram.percent:.1f}%) | "
                f"CPU: {psutil.cpu_percent():.1f}% ({psutil.cpu_count()} cores) | "
                f"Disk: {disk.used/1024**3:.1f}/{disk.total/1024**3:.1f}GB ({disk.used/disk.total*100:.1f}%){speed_info}")


def validate_dataset_samples(dataset, processor, logger, max_samples=5):
    """Validate dataset samples for token ID issues"""
    logger.info(f"Validating {min(max_samples, len(dataset))} dataset samples...")
    
    vocab_size = len(processor.tokenizer)
    model_vocab_size = getattr(processor.tokenizer, 'vocab_size', vocab_size)
    
    for i in range(min(max_samples, len(dataset))):
        try:
            sample = dataset[i]
            
            # Check input_ids if present
            if 'input_ids' in sample:
                input_ids = sample['input_ids']
                if isinstance(input_ids, torch.Tensor):
                    max_id = input_ids.max().item()
                    min_id = input_ids.min().item()
                    logger.info(f"Sample {i}: input_ids range [{min_id}, {max_id}], vocab_size: {vocab_size}")
                    
                    if max_id >= vocab_size:
                        logger.error(f"Invalid token ID {max_id} >= vocab_size {vocab_size}")
                        return False
                        
            # Check labels if present
            if 'labels' in sample:
                labels = sample['labels']
                if isinstance(labels, torch.Tensor):
                    # Filter out -100 (ignore index)
                    valid_labels = labels[labels != -100]
                    if len(valid_labels) > 0:
                        max_label = valid_labels.max().item()
                        min_label = valid_labels.min().item()
                        logger.info(f"Sample {i}: labels range [{min_label}, {max_label}] (excluding -100)")
                        
                        if max_label >= vocab_size:
                            logger.error(f"Invalid label ID {max_label} >= vocab_size {vocab_size}")
                            return False
                            
        except Exception as e:
            logger.warning(f"Error validating sample {i}: {e}")
            return False
    
    logger.info("Dataset validation passed!")
    return True


def check_corrupted_samples(raw_dataset, logger):
    """Check for corrupted samples in the raw dataset"""
    logger.info("Checking for corrupted samples...")
    
    corrupted_count = 0
    for split_name, split_data in raw_dataset.items():
        logger.info(f"Checking {split_name} split...")
        
        for i, sample in enumerate(split_data):
            if not sample.get('html_table') or not sample.get('image'):
                logger.warning(f"{split_name} sample {i} has missing html_table or image")
                corrupted_count += 1
                
            # Check for extremely long HTML
            html_content = sample.get('html_table', '')
            if len(html_content) > 50000:
                logger.warning(f"{split_name} sample {i} has very long HTML ({len(html_content)} chars)")
    
    logger.info(f"Found {corrupted_count} potentially corrupted samples")
    return corrupted_count


class TEDSEvaluator:
    """TEDS evaluator for table structure recognition"""
    
    def __init__(self, model, processor, template, logger):
        self.model = model
        self.processor = processor
        self.template = template
        self.logger = logger
        self.device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        
        # Initialize TEDS scorers
        self.teds_structure_content = TEDS(structure_only=False)
        self.teds_structure_only = TEDS(structure_only=True)
        
        self.logger.info("TEDS evaluators initialized (Structure+Content & Structure-only)")
    
    def generate_prediction(self, sample, max_new_tokens=8192):
        """Generate HTML prediction for a sample"""
        try:
            self.model.eval()
            with torch.no_grad():
                # Encode input
                inputs = self.template.encode({'messages': [sample['messages'][0]], 'images': sample.get('images', [])})
                input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Validate input_ids
                vocab_size = len(self.processor.tokenizer)
                if input_ids.max() >= vocab_size:
                    self.logger.warning(f"Invalid input token ID {input_ids.max()} >= vocab_size {vocab_size}")
                    return ""
                
                # Prepare generation kwargs
                generation_kwargs = {
                    'use_cache': False, 
                    'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                    'temperature': 0.1,
                    'do_sample': False
                }
                
                # Add other inputs
                for key, tensor_key in [('attention_mask', 'attention_mask'), ('pixel_values', 'pixel_values'), ('image_grid_thw', 'image_grid_thw')]:
                    if key in inputs:
                        tensor = (inputs[key].clone().detach() if isinstance(inputs[key], torch.Tensor) 
                                else torch.tensor(inputs[key])).to(self.device)
                        if tensor.dim() == 1 and key != 'pixel_values':
                            tensor = tensor.unsqueeze(0)
                        generation_kwargs[tensor_key] = tensor
                
                # Generate and decode
                outputs = self.model.generate(
                    input_ids, 
                    max_new_tokens=max_new_tokens, 
                    num_beams=1, 
                    eos_token_id=self.processor.tokenizer.eos_token_id, 
                    **generation_kwargs
                )
                
                prediction = self.processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                return prediction.strip()
                
        except Exception as e:
            self.logger.warning(f"Generation error: {e}")
            return ""
    
    def evaluate_samples(self, eval_dataset, max_samples):
        """Evaluate TEDS scores on a subset of samples"""
        self.logger.info(f"TEDS EVALUATION ON {min(max_samples, len(eval_dataset))} SAMPLES")
        
        structure_content_scores, structure_only_scores, successful = [], [], 0
        
        for i in range(min(max_samples, len(eval_dataset))):
            try:
                sample = eval_dataset[i]
                pred_html = self.generate_prediction(sample)
                
                # Get ground truth
                true_html = (sample['messages'][1]['content'] if 'messages' in sample and len(sample['messages']) > 1 
                           else sample.get('html_table', ''))
                
                if not pred_html.strip() or not true_html.strip():
                    self.logger.warning(f"Sample {i+1}: Empty HTML (pred: {len(pred_html)}, true: {len(true_html)})")
                    continue
                
                # Calculate TEDS scores
                teds_sc = self.teds_structure_content(true_html, pred_html)
                teds_so = self.teds_structure_only(true_html, pred_html)
                
                structure_content_scores.append(teds_sc)
                structure_only_scores.append(teds_so)
                successful += 1
                
                self.logger.info(f"Sample {i+1} TEDS - Structure+Content: {teds_sc:.4f}, Structure: {teds_so:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Sample {i+1}: Error: {e}")
        
        # Calculate averages
        avg_sc = np.mean(structure_content_scores) if structure_content_scores else 0.0
        avg_so = np.mean(structure_only_scores) if structure_only_scores else 0.0
        
        self.logger.info(f"TEDS RESULTS: {successful}/{max_samples} samples | Structure+Content: {avg_sc:.4f} | Structure: {avg_so:.4f}")
        
        return {'eval_teds_structure_content': avg_sc, 'eval_teds_structure_only': avg_so, 'eval_teds_samples': successful}


class ResourceCallback(TrainerCallback):
    """Enhanced callback for resource logging and TEDS evaluation"""
    def __init__(self, logger, teds_evaluator=None, eval_dataset=None):
        self.logger = logger
        self.teds_evaluator = teds_evaluator
        self.eval_dataset = eval_dataset
        self.last_time = self.last_step = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 10 == 0:
            # Calculate samples per second
            current_time = time.time()
            if self.last_time and state.global_step > self.last_step:
                time_elapsed = current_time - self.last_time
                steps_elapsed = state.global_step - self.last_step
                samples_per_sec = (steps_elapsed * args.per_device_train_batch_size * args.gradient_accumulation_steps) / time_elapsed
                log_resources(self.logger, f"Step-{state.global_step}", samples_per_sec)
            else:
                log_resources(self.logger, f"Step-{state.global_step}")
            
            self.last_time, self.last_step = current_time, state.global_step
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Run TEDS evaluation after training evaluation"""
        if not self.teds_evaluator or not self.eval_dataset:
            return
            
        if state.global_step >= 50:
            try:
                teds_results = self.teds_evaluator.evaluate_samples(self.eval_dataset, 3)
                self.logger.info(f"[Step {state.global_step}] TEDS: {teds_results['eval_teds_structure_content']:.4f} (SC) | "
                               f"{teds_results['eval_teds_structure_only']:.4f} (SO) | {teds_results['eval_teds_samples']} samples")
                
                # Log to tensorboard
                if hasattr(state, 'log_history'):
                    state.log_history[-1].update(teds_results)
            except Exception as e:
                self.logger.error(f"TEDS evaluation failed: {e}")
        else:
            self.logger.info(f"[Step {state.global_step}] Skipping TEDS - waiting for step 50+")


def main():
    """Main training function"""
    logger = get_logger()
    seed_everything(42)
    
    # Configuration
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
    output_dir = 'output'
    
    # Calculate training parameters
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gradient_accumulation_steps = max(8 // gpu_count, 1)  # Increased for better stability
    total_samples, epochs = 222, 6
    expected_steps = (total_samples * epochs) // (1 * gradient_accumulation_steps * gpu_count)
    
    logger.info(f"Training setup: {gpu_count} GPUs, grad_accum={gradient_accumulation_steps}, expected_steps={expected_steps}")
    
    # Training arguments with enhanced stability
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        num_train_epochs=epochs,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=42,
        remove_unused_columns=False,
        bf16=True,
        optim='adafactor',
        max_grad_norm=1.0,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        # Enhanced stability settings
        skip_memory_metrics=True,
        dataloader_pin_memory=False,
        ignore_data_skip=True,
    )
    
    log_resources(logger, "START")
    
    # Load model and setup
    logger.info("Loading model...")
    model, processor = get_model_tokenizer(model_id_or_path)
    
    # Enhanced template setup with validation
    template = get_template(model.model_meta.template, processor, max_length=32768)
    template.set_mode('train')
    if template.use_model:
        template.model = model
    
    # Log template and tokenizer info
    logger.info(f"Template info: {template}")
    logger.info(f"Tokenizer vocab size: {len(processor.tokenizer)}")
    logger.info(f"Model vocab size: {getattr(model.config, 'vocab_size', 'N/A')}")
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    target_modules = get_multimodal_target_regex(model, freeze_llm=False, freeze_vit=True, freeze_aligner=True)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=4, lora_alpha=16, target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    
    logger.info(f'Model: {get_model_parameter_info(model)}')
    
    # Load and process dataset with validation
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Check for corrupted samples
    check_corrupted_samples(raw_dataset, logger)
    
    # Process dataset and filter out None samples
    logger.info("Processing dataset...")
    train_processed_raw = [create_swift_format_single(sample) for sample in raw_dataset['train']]
    val_processed_raw = [create_swift_format_single(sample) for sample in raw_dataset['validation']]
    
    # Filter out None samples
    train_processed = [s for s in train_processed_raw if s is not None]
    val_processed = [s for s in val_processed_raw if s is not None]
    
    logger.info(f"Dataset processing: {len(train_processed)}/{len(train_processed_raw)} train samples kept")
    logger.info(f"Dataset processing: {len(val_processed)}/{len(val_processed_raw)} val samples kept")
    
    # Create LazyLLMDataset
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=42)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=42)
    
    logger.info(f"Final dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    
    # Validate dataset samples
    if not validate_dataset_samples(train_dataset, processor, logger, max_samples=5):
        logger.error("Dataset validation failed! Stopping training.")
        return
    
    # Initialize TEDS evaluator
    teds_evaluator = TEDSEvaluator(model, processor, template, logger)
    
    # Memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Train with enhanced error handling
    logger.info("Starting training...")
    model.enable_input_require_grads()
    
    try:
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
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    
    # Final evaluation
    logger.info("Final TEDS evaluation...")
    try:
        final_results = teds_evaluator.evaluate_samples(val_processed, 10)
        logger.info(f'Final TEDS - Structure+Content: {final_results["eval_teds_structure_content"]:.4f} | '
                    f'Structure: {final_results["eval_teds_structure_only"]:.4f}')
    except Exception as e:
        logger.error(f"Final evaluation failed: {e}")
    
    log_resources(logger, "END")
    
    # Save visualization
    try:
        visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
        os.makedirs(visual_loss_dir, exist_ok=True)
        plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    except Exception as e:
        logger.warning(f"Failed to save visualization: {e}")
    
    logger.info(f'Training complete. Best model: {getattr(trainer.state, "best_model_checkpoint", "N/A")}')


if __name__ == "__main__":
    main()