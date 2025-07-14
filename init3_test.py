#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script with Optimized Multi-GPU Distribution (Streamlined)
"""

import os
import psutil
import torch
import time
import numpy as np
from pathlib import Path

from swift.llm import get_model_tokenizer, get_template, get_multimodal_target_regex, LazyLLMDataset
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset as hf_load_dataset
from transformers import TrainerCallback

try:
    from table_recognition_metric import TEDS
    print("TEDS library loaded successfully")
except ImportError:
    print("TEDS library not found. Install with: pip install table-recognition-metric")
    raise


class GPUManager:
    """Centralized GPU management and monitoring"""
    
    def __init__(self, logger):
        self.logger = logger
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._gpu_info_cache = {}
        self._cache_time = 0
        
        if self.gpu_count > 0:
            self._setup_multi_gpu()
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU environment"""
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(self.gpu_count))
        
        if self.gpu_count > 1:
            os.environ.update({
                'WORLD_SIZE': str(self.gpu_count),
                'MASTER_ADDR': 'localhost',
                'MASTER_PORT': '12355',
                'NCCL_SOCKET_IFNAME': 'lo'
            })
            self.logger.info(f"Configured distributed training for {self.gpu_count} GPUs")
        
        # Set memory fraction and clear cache
        for i in range(self.gpu_count):
            torch.cuda.set_per_process_memory_fraction(0.85, device=i)
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
    
    def get_gpu_info(self, force_refresh=False):
        """Get GPU info with caching"""
        current_time = time.time()
        if force_refresh or current_time - self._cache_time > 2:  # Cache for 2 seconds
            self._gpu_info_cache = {}
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                total = props.total_memory
                self._gpu_info_cache[i] = {
                    'allocated': allocated / 1024**3,
                    'total': total / 1024**3,
                    'free': (total - torch.cuda.memory_reserved(i)) / 1024**3,
                    'utilization': (allocated / total) * 100
                }
            self._cache_time = current_time
        return self._gpu_info_cache
    
    def get_best_gpu(self):
        """Get GPU with most free memory"""
        if self.gpu_count <= 1:
            return 0
        gpu_info = self.get_gpu_info()
        return max(gpu_info.keys(), key=lambda x: gpu_info[x]['free'])
    
    def log_status(self, step="", samples_per_sec=None):
        """Log GPU and system status"""
        if self.gpu_count > 0:
            gpu_info = self.get_gpu_info()
            gpu_summary = [f"GPU{i}: {info['allocated']:.1f}/{info['total']:.1f}GB ({info['utilization']:.1f}%)" 
                          for i, info in gpu_info.items()]
            avg_util = sum(info['utilization'] for info in gpu_info.values()) / len(gpu_info)
            self.logger.info(f"[{step}] {' | '.join(gpu_summary)} | Avg: {avg_util:.1f}%")
        
        # System resources
        ram = psutil.virtual_memory()
        speed_info = f" | Speed: {samples_per_sec:.3f} samples/s" if samples_per_sec else ""
        self.logger.info(f"[{step}] RAM: {ram.used/1024**3:.1f}/{ram.total/1024**3:.1f}GB | "
                        f"CPU: {psutil.cpu_percent():.1f}%{speed_info}")


class TEDSEvaluator:
    """Unified TEDS evaluator with automatic GPU distribution"""
    
    def __init__(self, model, processor, template, gpu_manager, logger):
        self.model = model
        self.processor = processor
        self.template = template
        self.gpu_manager = gpu_manager
        self.logger = logger
        
        # Initialize TEDS scorers
        self.teds_structure_content = TEDS(structure_only=False)
        self.teds_structure_only = TEDS(structure_only=True)
        
        self.logger.info("TEDS evaluators initialized")
    
    def _prepare_inputs(self, sample, device):
        """Prepare model inputs for generation"""
        inputs = self.template.encode({'messages': [sample['messages'][0]], 'images': sample.get('images', [])})
        
        generation_kwargs = {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long, device=device).unsqueeze(0),
            'use_cache': False,
            'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
            'do_sample': False,
            'num_beams': 1,
            'eos_token_id': self.processor.tokenizer.eos_token_id
        }
        
        # Add other inputs
        for key, tensor_key in [('attention_mask', 'attention_mask'), ('pixel_values', 'pixel_values'), ('image_grid_thw', 'image_grid_thw')]:
            if key in inputs:
                tensor = (inputs[key].clone().detach() if isinstance(inputs[key], torch.Tensor) 
                         else torch.tensor(inputs[key])).to(device)
                if tensor.dim() == 1 and key != 'pixel_values':
                    tensor = tensor.unsqueeze(0)
                generation_kwargs[tensor_key] = tensor
        
        return generation_kwargs
    
    def generate_prediction(self, sample, max_new_tokens=8192):
        """Generate HTML prediction with optimal GPU selection"""
        try:
            device = f'cuda:{self.gpu_manager.get_best_gpu()}' if self.gpu_manager.gpu_count > 0 else 'cpu'
            
            self.model.eval()
            with torch.no_grad():
                generation_kwargs = self._prepare_inputs(sample, device)
                input_ids = generation_kwargs.pop('input_ids')
                
                outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, **generation_kwargs)
                prediction = self.processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                
                # Clear cache only when using GPU
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                
                return prediction.strip()
                
        except Exception as e:
            self.logger.warning(f"Generation error: {e}")
            return ""
    
    def evaluate_samples(self, eval_dataset, max_samples):
        """Evaluate TEDS scores on samples"""
        self.logger.info(f"TEDS EVALUATION ON {min(max_samples, len(eval_dataset))} SAMPLES")
        
        structure_content_scores, structure_only_scores, successful = [], [], 0
        
        for i in range(min(max_samples, len(eval_dataset))):
            try:
                sample = eval_dataset[i]
                pred_html = self.generate_prediction(sample)
                
                true_html = (sample['messages'][1]['content'] if 'messages' in sample and len(sample['messages']) > 1 
                           else sample.get('html_table', ''))
                
                if not pred_html.strip() or not true_html.strip():
                    continue
                
                teds_sc = self.teds_structure_content(true_html, pred_html)
                teds_so = self.teds_structure_only(true_html, pred_html)
                
                structure_content_scores.append(teds_sc)
                structure_only_scores.append(teds_so)
                successful += 1
                
                self.logger.info(f"Sample {i+1} TEDS - Structure+Content: {teds_sc:.4f}, Structure: {teds_so:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Sample {i+1}: Error: {e}")
        
        avg_sc = np.mean(structure_content_scores) if structure_content_scores else 0.0
        avg_so = np.mean(structure_only_scores) if structure_only_scores else 0.0
        
        self.logger.info(f"TEDS RESULTS: {successful}/{max_samples} samples | "
                        f"Structure+Content: {avg_sc:.4f} | Structure: {avg_so:.4f}")
        
        return {'eval_teds_structure_content': avg_sc, 'eval_teds_structure_only': avg_so, 'eval_teds_samples': successful}


class TrainingCallback(TrainerCallback):
    """Streamlined training callback"""
    
    def __init__(self, gpu_manager, teds_evaluator=None, eval_dataset=None):
        self.gpu_manager = gpu_manager
        self.teds_evaluator = teds_evaluator
        self.eval_dataset = eval_dataset
        self.last_time = self.last_step = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 10 == 0:
            # Calculate speed
            samples_per_sec = None
            current_time = time.time()
            if self.last_time and state.global_step > self.last_step:
                time_elapsed = current_time - self.last_time
                steps_elapsed = state.global_step - self.last_step
                effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, self.gpu_manager.gpu_count)
                samples_per_sec = (steps_elapsed * effective_batch_size) / time_elapsed
            
            self.gpu_manager.log_status(f"Step-{state.global_step}", samples_per_sec)
            self.last_time, self.last_step = current_time, state.global_step
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if self.teds_evaluator and self.eval_dataset and state.global_step >= 50:
            try:
                teds_results = self.teds_evaluator.evaluate_samples(self.eval_dataset, 3)
                if hasattr(state, 'log_history'):
                    state.log_history[-1].update(teds_results)
            except Exception as e:
                self.gpu_manager.logger.error(f"TEDS evaluation failed: {e}")


def create_swift_format_single(sample):
    """Convert single sample to Swift format"""
    return {
        'messages': [
            {'role': 'user', 'content': 'Write the HTML representation for this image of a medical table.'},
            {'role': 'assistant', 'content': sample['html_table']}
        ],
        'images': [sample['image']]
    }


def main():
    """Main training function"""
    logger = get_logger()
    seed_everything(42)
    
    # Initialize GPU manager
    gpu_manager = GPUManager(logger)
    
    # Configuration
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
    output_dir = 'output'
    
    # Calculate training parameters
    effective_gpu_count = max(1, gpu_manager.gpu_count)
    gradient_accumulation_steps = max(4 // effective_gpu_count, 1)
    per_device_batch_size = 1
    
    logger.info(f"Training setup: {effective_gpu_count} GPUs, batch_size={per_device_batch_size}, "
                f"grad_accum={gradient_accumulation_steps}")
    
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
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=50,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=6,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=min(4, os.cpu_count()),
        data_seed=42,
        remove_unused_columns=False,
        bf16=True,
        optim='adafactor',
        max_grad_norm=1.0,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        ddp_backend='nccl' if gpu_manager.gpu_count > 1 else None,
        dataloader_pin_memory=True,
    )
    
    gpu_manager.log_status("START")
    
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
    
    # Load dataset
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    train_processed = raw_dataset['train'].map(create_swift_format_single)
    val_processed = raw_dataset['validation'].map(create_swift_format_single)
    
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=42)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=42)
    
    logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
    
    # Initialize TEDS evaluator
    teds_evaluator = TEDSEvaluator(model, processor, template, gpu_manager, logger)
    
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
        callbacks=[TrainingCallback(gpu_manager, teds_evaluator, val_processed)],
    )
    
    trainer.train()
    
    # Final evaluation
    logger.info("Final TEDS evaluation...")
    final_results = teds_evaluator.evaluate_samples(val_processed, 10)
    
    gpu_manager.log_status("END")
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Best model: {trainer.state.best_model_checkpoint}')
    logger.info(f'Final TEDS - Structure+Content: {final_results["eval_teds_structure_content"]:.4f} | '
                f'Structure: {final_results["eval_teds_structure_only"]:.4f}')


if __name__ == "__main__":
    main()