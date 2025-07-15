#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script for Table HTML Conversion with TEDS Evaluation
Enhanced with multi-GPU optimizations and advanced monitoring
"""

import os
import psutil
import torch
import time
import numpy as np
import gc

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

# Optional GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
    print("pynvml loaded for enhanced GPU monitoring")
except ImportError:
    PYNVML_AVAILABLE = False
    print("pynvml not available - install with: pip install pynvml")


def create_swift_format_single(sample):
    """Convert single sample to Swift format"""
    return {
        'messages': [
            {'role': 'user', 'content': 'Write the HTML representation for this image of a medical table.'},
            {'role': 'assistant', 'content': sample['html_table']}
        ],
        'images': [sample['image']]
    }


def calculate_training_params(gpu_count, logger):
    """Calculate optimal training parameters based on GPU count"""
    # Configuration
    min_effective_batch = 8      # Minimum for training stability
    target_per_device_batch = 2  # Preferred batch size per GPU
    max_memory_per_device = 4    # Maximum samples per device (memory constraint)
    
    # Start with target batch size
    per_device_batch = min(target_per_device_batch, max_memory_per_device)
    
    # Calculate required gradient accumulation
    base_effective = per_device_batch * gpu_count
    
    if base_effective >= min_effective_batch:
        # We can achieve minimum with current setup
        gradient_accumulation_steps = 1
    else:
        # Need gradient accumulation to reach minimum
        gradient_accumulation_steps = max(1, min_effective_batch // base_effective)
    
    # Check if we exceed memory constraints
    total_memory_per_device = per_device_batch * gradient_accumulation_steps
    if total_memory_per_device > max_memory_per_device:
        # Reduce per-device batch and increase gradient accumulation
        per_device_batch = 1
        gradient_accumulation_steps = max(1, min_effective_batch // gpu_count)
    
    effective_batch = per_device_batch * gpu_count * gradient_accumulation_steps
    
    logger.info(f"Training config: {per_device_batch} per-device × {gpu_count} GPUs × {gradient_accumulation_steps} accum = {effective_batch} effective")
    
    return per_device_batch, gradient_accumulation_steps, effective_batch


def get_gpu_info(gpu_id):
    """Get comprehensive GPU information for a single GPU"""
    if not torch.cuda.is_available():
        return None
    
    try:
        # Memory stats
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        
        gpu_info = {
            'allocated': allocated,
            'reserved': reserved,
            'total': total_mem,
            'fragmentation': (reserved - allocated) / total_mem * 100 if total_mem > 0 else 0
        }
        
        # Enhanced metrics (if pynvml available)
        if PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info['utilization'] = utilization.gpu
                
                # Temperature
                gpu_info['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power (with safer API call)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    gpu_info['power'] = power
                    # Try to get power limit, fallback if not available
                    try:
                        power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                        gpu_info['power_limit'] = power_limit
                    except:
                        gpu_info['power_limit'] = None
                except:
                    gpu_info['power'] = None
                    gpu_info['power_limit'] = None
                    
            except Exception:
                gpu_info.update({'utilization': None, 'temperature': None, 'power': None, 'power_limit': None})
        else:
            gpu_info.update({'utilization': None, 'temperature': None, 'power': None, 'power_limit': None})
        
        return gpu_info
        
    except Exception:
        return None


def log_resources(logger, step="", samples_per_sec=None):
    """Enhanced resource logging with detailed GPU metrics"""
    
    # GPU detailed monitoring
    if torch.cuda.is_available():
        gpu_details = []
        total_allocated = 0
        total_capacity = 0
        
        for i in range(torch.cuda.device_count()):
            gpu_info = get_gpu_info(i)
            if gpu_info is None:
                continue
                
            total_allocated += gpu_info['allocated']
            total_capacity += gpu_info['total']
            
            # Format GPU info string
            util_str = f"{gpu_info['utilization']}%" if gpu_info['utilization'] is not None else "N/A"
            temp_str = f"{gpu_info['temperature']}°C" if gpu_info['temperature'] is not None else "N/A"
            
            if gpu_info['power'] is not None:
                if gpu_info['power_limit'] is not None:
                    power_str = f"{gpu_info['power']:.0f}/{gpu_info['power_limit']:.0f}W"
                else:
                    power_str = f"{gpu_info['power']:.0f}W"
            else:
                power_str = "N/A"
            
            gpu_details.append(
                f"GPU{i}: {gpu_info['allocated']:.1f}/{gpu_info['total']:.1f}GB "
                f"(Util:{util_str}, Frag:{gpu_info['fragmentation']:.1f}%, "
                f"Temp:{temp_str}, Power:{power_str})"
            )
        
        gpu_summary = " | ".join(gpu_details)
        avg_utilization = total_allocated / total_capacity * 100 if total_capacity > 0 else 0
        
    else:
        gpu_summary = "N/A"
        avg_utilization = 0
    
    # System resources
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Network I/O (with specific exception handling)
    try:
        net_io = psutil.net_io_counters()
        if net_io:
            network_str = f"Net: {net_io.bytes_sent/1024**3:.1f}GB↑/{net_io.bytes_recv/1024**3:.1f}GB↓"
        else:
            network_str = "Net: N/A"
    except (AttributeError, OSError):
        network_str = "Net: N/A"
    
    # Training speed
    speed_info = f" | Speed: {samples_per_sec:.3f} samples/s" if samples_per_sec else ""
    
    # Consolidated logging
    if len(gpu_details) <= 2:  # Single line for 1-2 GPUs
        logger.info(f"[{step}] {gpu_summary}")
    else:  # Multi-line for many GPUs
        logger.info(f"[{step}] GPUs: {len(gpu_details)} total")
        for detail in gpu_details:
            logger.info(f"[{step}]   {detail}")
    
    logger.info(f"[{step}] Overall GPU: {avg_utilization:.1f}% | "
                f"RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f}GB ({ram.percent:.1f}%) | "
                f"CPU: {psutil.cpu_percent():.1f}% ({psutil.cpu_count()} cores) | "
                f"Disk: {disk.used/1024**3:.1f}/{disk.total/1024**3:.1f}GB | "
                f"{network_str}{speed_info}")


def log_memory_breakdown(logger, step=""):
    """Detailed memory breakdown for debugging - simplified to avoid redundancy"""
    if not torch.cuda.is_available():
        return
        
    logger.info(f"[{step}] MEMORY BREAKDOWN:")
    
    for i in range(torch.cuda.device_count()):
        gpu_info = get_gpu_info(i)
        if gpu_info is None:
            continue
            
        # Get peak memory
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        
        logger.info(f"  GPU {i}: Current={gpu_info['allocated']:.1f}GB, "
                   f"Reserved={gpu_info['reserved']:.1f}GB, Peak={max_allocated:.1f}GB")
        
        # Reset peak memory stats for next measurement
        torch.cuda.reset_peak_memory_stats(i)


class TEDSEvaluator:
    """TEDS evaluator for table structure recognition with enhanced multi-GPU support"""
    
    def __init__(self, model, processor, template, logger):
        # Enhanced device management for multi-GPU
        if hasattr(model, 'module'):  # DDP wrapped model
            self.model = model.module  # Get the actual model
            # For DDP, get device from first parameter (they should all be on same device per process)
            try:
                self.device = next(iter(model.module.parameters())).device
                logger.info(f"TEDS: Using DDP-wrapped model on device: {self.device}")
            except StopIteration:
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                logger.warning(f"TEDS: No parameters found, defaulting to device: {self.device}")
        else:  # Single GPU or non-DDP
            self.model = model
            try:
                self.device = next(iter(model.parameters())).device
                logger.info(f"TEDS: Using single GPU device: {self.device}")
            except StopIteration:
                self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                logger.warning(f"TEDS: No parameters found, defaulting to device: {self.device}")
        
        self.processor = processor
        self.template = template
        self.logger = logger
        
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
                
                # Prepare generation kwargs
                generation_kwargs = {'use_cache': False, 'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id}
                
                # Add other inputs
                for key, tensor_key in [('attention_mask', 'attention_mask'), ('pixel_values', 'pixel_values'), ('image_grid_thw', 'image_grid_thw')]:
                    if key in inputs:
                        tensor = (inputs[key].clone().detach() if isinstance(inputs[key], torch.Tensor) 
                                else torch.tensor(inputs[key])).to(self.device)
                        if tensor.dim() == 1 and key != 'pixel_values':
                            tensor = tensor.unsqueeze(0)
                        generation_kwargs[tensor_key] = tensor
                
                # Generate and decode
                outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, num_beams=1, 
                                            do_sample=False, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_kwargs)
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


class MemoryCallback(TrainerCallback):
    """Memory-aware training callback with automatic cleanup"""
    def __init__(self, logger, memory_threshold=0.85):
        self.logger = logger
        self.memory_threshold = memory_threshold
        self.last_memory_check = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        # Only check memory every 25 steps to reduce overhead
        if state.global_step % 25 == 0 and state.global_step != self.last_memory_check:
            self.last_memory_check = state.global_step
            
            memory_warnings = []
            cleanup_needed = False
            
            for i in range(torch.cuda.device_count()):
                gpu_info = get_gpu_info(i)
                if gpu_info is None:
                    continue
                    
                usage_pct = gpu_info['allocated'] / gpu_info['total']
                
                if usage_pct > self.memory_threshold:
                    memory_warnings.append(f"GPU {i}: {usage_pct:.1%}")
                    cleanup_needed = True
            
            if cleanup_needed:
                self.logger.warning(f"High memory usage detected: {', '.join(memory_warnings)}")
                
                # Trigger cleanup
                gc.collect()
                torch.cuda.empty_cache()
                self.logger.info("Performed memory cleanup (gc + cache clear)")
            
            # Detailed breakdown every 100 steps (less frequent than before)
            if state.global_step % 100 == 0:
                log_memory_breakdown(self.logger, f"Step-{state.global_step}")


def main():
    """Main training function"""
    logger = get_logger()
    seed_everything(42)
    
    # Configuration
    model_id_or_path = 'Qwen/Qwen2.5-VL-32B-Instruct'
    output_dir = 'output'
    
    # Calculate training parameters with smart GPU scaling
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    per_device_batch, gradient_accumulation_steps, effective_batch = calculate_training_params(gpu_count, logger)
    
    total_samples, epochs = 222, 6
    expected_steps = (total_samples * epochs) // effective_batch
    
    logger.info(f"Training setup: {gpu_count} GPUs, effective_batch={effective_batch}, expected_steps={expected_steps}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
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
    )
    
    log_resources(logger, "START")
    
    # Load model and setup
    logger.info("Loading model...")
    model, processor = get_model_tokenizer(model_id_or_path)
    template = get_template(model.model_meta.template, processor, max_length=32768)
    template.set_mode('train')
    if template.use_model:
        template.model = model
    
    # Setup LoRA
    logger.info("Setting up LoRA...")
    target_modules = get_multimodal_target_regex(model, freeze_llm=False, freeze_vit=True, freeze_aligner=True)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=4, lora_alpha=16, target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    
    logger.info(f'Model: {get_model_parameter_info(model)}')
    
    # Load and process dataset
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Filter out samples with HTML longer than 15,000 characters
    def filter_long_html(sample):
        """Filter function to remove samples with long HTML"""
        html_content = sample.get('html_table', '')
        return len(html_content) <= 15000
    
    logger.info("Filtering long HTML samples...")
    train_filtered = raw_dataset['train'].filter(filter_long_html)
    val_filtered = raw_dataset['validation'].filter(filter_long_html)
    
    logger.info(f"Original dataset: {len(raw_dataset['train'])} train, {len(raw_dataset['validation'])} validation")
    logger.info(f"Filtered dataset: {len(train_filtered)} train, {len(val_filtered)} validation samples")
    
    train_processed = train_filtered.map(create_swift_format_single)
    val_processed = val_filtered.map(create_swift_format_single)
    
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=42)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=42)
    
    logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    
    # Initialize TEDS evaluator
    teds_evaluator = TEDSEvaluator(model, processor, template, logger)
    
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
        callbacks=[
            ResourceCallback(logger, teds_evaluator, val_processed),
            MemoryCallback(logger, memory_threshold=0.85)
        ],
    )
    
    trainer.train()
    
    # Final evaluation
    logger.info("Final TEDS evaluation...")
    final_results = teds_evaluator.evaluate_samples(val_processed, 10)
    
    log_resources(logger, "END")
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Best model: {trainer.state.best_model_checkpoint}')
    logger.info(f'Final TEDS - Structure+Content: {final_results["eval_teds_structure_content"]:.4f} | '
                f'Structure: {final_results["eval_teds_structure_only"]:.4f}')


if __name__ == "__main__":
    main()