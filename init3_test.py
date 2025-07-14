#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script with Optimized Parallel TEDS Evaluation
"""

import os
import psutil
import torch
import time
import numpy as np
import concurrent.futures
import threading
from functools import partial

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
    """Convert single sample to Swift format"""
    return {
        'messages': [
            {'role': 'user', 'content': 'Write the HTML representation for this image of a medical table.'},
            {'role': 'assistant', 'content': sample['html_table']}
        ],
        'images': [sample['image']]
    }


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


class OptimizedTEDSEvaluator:
    """TEDS evaluator with parallel processing across multiple GPUs"""
    
    def __init__(self, model, processor, template, logger):
        self.model = model
        self.processor = processor
        self.template = template
        self.logger = logger
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Initialize TEDS scorers (these are thread-safe)
        self.teds_structure_content = TEDS(structure_only=False)
        self.teds_structure_only = TEDS(structure_only=True)
        
        # Thread-local storage for GPU contexts
        self.thread_local = threading.local()
        
        self.logger.info(f"Optimized TEDS evaluator initialized with {self.gpu_count} GPUs")
    
    def _get_gpu_context(self, gpu_id):
        """Get or create GPU context for current thread"""
        if not hasattr(self.thread_local, 'gpu_contexts'):
            self.thread_local.gpu_contexts = {}
        
        if gpu_id not in self.thread_local.gpu_contexts:
            self.thread_local.gpu_contexts[gpu_id] = torch.cuda.device(gpu_id)
        
        return self.thread_local.gpu_contexts[gpu_id]
    
    def generate_prediction_on_gpu(self, sample, gpu_id, max_new_tokens=2048):
        """Generate HTML prediction for a sample on specific GPU"""
        try:
            # Set GPU context for this thread
            with self._get_gpu_context(gpu_id):
                # Set the current device
                torch.cuda.set_device(gpu_id)
                
                # Ensure model is in eval mode
                self.model.eval()
                
                with torch.no_grad():
                    # Encode input
                    inputs = self.template.encode({
                        'messages': [sample['messages'][0]], 
                        'images': sample.get('images', [])
                    })
                    
                    # Move inputs to current GPU
                    input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long, device=f'cuda:{gpu_id}').unsqueeze(0)
                    
                    # Prepare generation kwargs
                    generation_kwargs = {
                        'use_cache': False, 
                        'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                        'do_sample': False,
                        'early_stopping': True,
                    }
                    
                    # Add other inputs and move to GPU
                    for key, tensor_key in [('attention_mask', 'attention_mask'), 
                                          ('pixel_values', 'pixel_values'), 
                                          ('image_grid_thw', 'image_grid_thw')]:
                        if key in inputs:
                            tensor = (inputs[key].clone().detach() if isinstance(inputs[key], torch.Tensor) 
                                    else torch.tensor(inputs[key])).to(f'cuda:{gpu_id}')
                            if tensor.dim() == 1 and key != 'pixel_values':
                                tensor = tensor.unsqueeze(0)
                            generation_kwargs[tensor_key] = tensor
                    
                    # Generate
                    outputs = self.model.generate(
                        input_ids, 
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        **generation_kwargs
                    )
                    
                    # Decode prediction
                    prediction = self.processor.tokenizer.decode(
                        outputs[0][input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    # Clear GPU cache after generation
                    torch.cuda.empty_cache()
                    
                    return prediction.strip()
                    
        except Exception as e:
            self.logger.warning(f"Generation error on GPU {gpu_id}: {e}")
            return ""
    
    def process_samples_on_gpu(self, samples_with_indices, gpu_id):
        """Process multiple samples on a specific GPU"""
        results = []
        
        self.logger.info(f"GPU {gpu_id}: Processing {len(samples_with_indices)} samples")
        
        for idx, sample in samples_with_indices:
            try:
                # Generate prediction
                pred_html = self.generate_prediction_on_gpu(sample, gpu_id, max_new_tokens=2048)
                
                # Get ground truth
                true_html = (sample['messages'][1]['content'] if 'messages' in sample and len(sample['messages']) > 1 
                           else sample.get('html_table', ''))
                
                if not pred_html.strip() or not true_html.strip():
                    self.logger.warning(f"GPU {gpu_id} Sample {idx+1}: Empty HTML")
                    continue
                
                # Calculate TEDS scores
                teds_sc = self.teds_structure_content(true_html, pred_html)
                teds_so = self.teds_structure_only(true_html, pred_html)
                
                results.append({
                    'sample_idx': idx,
                    'gpu_id': gpu_id,
                    'teds_structure_content': teds_sc,
                    'teds_structure_only': teds_so,
                    'pred_length': len(pred_html),
                    'true_length': len(true_html)
                })
                
                self.logger.info(f"GPU {gpu_id} Sample {idx+1}: TEDS SC={teds_sc:.4f}, SO={teds_so:.4f}")
                
            except Exception as e:
                self.logger.warning(f"GPU {gpu_id} Sample {idx+1}: Error: {e}")
        
        self.logger.info(f"GPU {gpu_id}: Completed {len(results)} samples successfully")
        return results
    
    def evaluate_samples_parallel(self, eval_dataset, max_samples):
        """Evaluate TEDS scores using parallel processing across GPUs"""
        start_time = time.time()
        actual_samples = min(max_samples, len(eval_dataset))
        
        self.logger.info(f"PARALLEL TEDS EVALUATION ON {actual_samples} SAMPLES ACROSS {self.gpu_count} GPUs")
        
        # Split samples across GPUs
        samples_per_gpu = actual_samples // self.gpu_count
        remaining_samples = actual_samples % self.gpu_count
        
        sample_chunks = []
        start_idx = 0
        
        for gpu_id in range(self.gpu_count):
            # Distribute remaining samples among first few GPUs
            chunk_size = samples_per_gpu + (1 if gpu_id < remaining_samples else 0)
            
            if chunk_size > 0:
                end_idx = start_idx + chunk_size
                samples_chunk = [(i, eval_dataset[i]) for i in range(start_idx, end_idx)]
                sample_chunks.append((gpu_id, samples_chunk))
                start_idx = end_idx
        
        # Log distribution
        for gpu_id, samples_chunk in sample_chunks:
            self.logger.info(f"GPU {gpu_id}: Assigned {len(samples_chunk)} samples")
        
        # Process in parallel using ThreadPoolExecutor
        all_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            # Submit tasks to thread pool
            futures = {
                executor.submit(self.process_samples_on_gpu, samples_chunk, gpu_id): gpu_id 
                for gpu_id, samples_chunk in sample_chunks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                gpu_id = futures[future]
                try:
                    gpu_results = future.result()
                    all_results.extend(gpu_results)
                    self.logger.info(f"GPU {gpu_id}: Completed successfully with {len(gpu_results)} results")
                except Exception as e:
                    self.logger.error(f"GPU {gpu_id}: Failed with error: {e}")
        
        # Calculate aggregate metrics
        if all_results:
            structure_content_scores = [r['teds_structure_content'] for r in all_results]
            structure_only_scores = [r['teds_structure_only'] for r in all_results]
            
            avg_sc = np.mean(structure_content_scores)
            avg_so = np.mean(structure_only_scores)
            successful_samples = len(all_results)
        else:
            avg_sc = avg_so = 0.0
            successful_samples = 0
        
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"PARALLEL TEDS RESULTS: {successful_samples}/{actual_samples} samples | "
                        f"Structure+Content: {avg_sc:.4f} | Structure: {avg_so:.4f} | "
                        f"Time: {elapsed_time:.1f}s | Speed: {successful_samples/elapsed_time:.2f} samples/s")
        
        return {
            'eval_teds_structure_content': avg_sc, 
            'eval_teds_structure_only': avg_so, 
            'eval_teds_samples': successful_samples,
            'eval_teds_time': elapsed_time
        }


class OptimizedResourceCallback(TrainerCallback):
    """Enhanced callback with optimized parallel TEDS evaluation"""
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
        """Run parallel TEDS evaluation after training evaluation"""
        if not self.teds_evaluator or not self.eval_dataset:
            return
            
        if state.global_step >= 100:  # Start evaluation later to save time
            try:
                # Use parallel evaluation with more samples since it's faster
                teds_results = self.teds_evaluator.evaluate_samples_parallel(self.eval_dataset, 8)
                
                self.logger.info(f"[Step {state.global_step}] PARALLEL TEDS: "
                               f"{teds_results['eval_teds_structure_content']:.4f} (SC) | "
                               f"{teds_results['eval_teds_structure_only']:.4f} (SO) | "
                               f"{teds_results['eval_teds_samples']} samples in {teds_results['eval_teds_time']:.1f}s")
                
                # Log to tensorboard
                if hasattr(state, 'log_history'):
                    state.log_history[-1].update(teds_results)
                    
            except Exception as e:
                self.logger.error(f"Parallel TEDS evaluation failed: {e}")
        else:
            self.logger.info(f"[Step {state.global_step}] Skipping TEDS - waiting for step 100+")


def main():
    """Main training function with optimizations"""
    logger = get_logger()
    seed_everything(42)
    
    # Configuration
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
    output_dir = 'output'
    
    # Calculate training parameters
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gradient_accumulation_steps = max(4 // gpu_count, 1)
    total_samples, epochs = 222, 6
    expected_steps = (total_samples * epochs) // (1 * gradient_accumulation_steps * gpu_count)
    
    logger.info(f"Optimized training setup: {gpu_count} GPUs, grad_accum={gradient_accumulation_steps}, expected_steps={expected_steps}")
    
    # Optimized training arguments
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
        save_steps=200,  # OPTIMIZED: Less frequent saving
        eval_strategy='steps',
        eval_steps=200,  # OPTIMIZED: Less frequent evaluation
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=3,  # OPTIMIZED: Keep fewer checkpoints
        logging_steps=5,
        dataloader_num_workers=2,  # OPTIMIZED: More data loading workers
        data_seed=42,
        remove_unused_columns=False,
        bf16=True,
        optim='adafactor',
        max_grad_norm=1.0,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,  # OPTIMIZED: Pin memory for faster transfer
    )
    
    log_resources(logger, "START")
    
    # Load model and setup
    logger.info("Loading model...")
    model, processor = get_model_tokenizer(model_id_or_path)
    template = get_template(model.model_meta.template, processor, max_length=16384)  # OPTIMIZED: Reduced max length
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
    train_processed = raw_dataset['train'].map(create_swift_format_single)
    val_processed = raw_dataset['validation'].map(create_swift_format_single)
    
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=42)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=42)
    
    logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    
    # Initialize optimized TEDS evaluator
    teds_evaluator = OptimizedTEDSEvaluator(model, processor, template, logger)
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train
    logger.info("Starting optimized training...")
    model.enable_input_require_grads()
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        callbacks=[OptimizedResourceCallback(logger, teds_evaluator, val_processed)],
    )
    
    trainer.train()
    
    # Final comprehensive evaluation
    logger.info("Final comprehensive TEDS evaluation...")
    final_results = teds_evaluator.evaluate_samples_parallel(val_processed, 20)  # More samples for final eval
    
    log_resources(logger, "END")
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Best model: {trainer.state.best_model_checkpoint}')
    logger.info(f'Final TEDS - Structure+Content: {final_results["eval_teds_structure_content"]:.4f} | '
                f'Structure: {final_results["eval_teds_structure_only"]:.4f} | '
                f'Time: {final_results["eval_teds_time"]:.1f}s')


if __name__ == "__main__":
    main()