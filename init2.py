#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script for Table HTML Conversion
"""

import os
import psutil
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset as hf_load_dataset
from transformers import TrainerCallback


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


def log_resources(logger, step="", samples_per_sec=None):
    """Log system resources and training metrics"""
    # GPU
    gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else "N/A"
    
    # System resources
    ram_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent()
    disk_percent = psutil.disk_usage('/').percent
    
    # Format training speed
    speed_info = f" | Speed: {samples_per_sec:.3f} samples/s" if samples_per_sec else ""
    
    logger.info(f"[{step}] GPU: {gpu_mem:.1f}GB ({gpu_util}%) | RAM: {ram_percent:.1f}% | "
                f"CPU: {cpu_percent:.1f}% | Disk: {disk_percent:.1f}%{speed_info}")


class ResourceCallback(TrainerCallback):
    """Minimal callback for resource logging"""
    def __init__(self, logger):
        self.logger = logger
        self.last_time = None
        self.last_step = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 10 == 0:  # Log every 10 steps
            # Calculate samples per second
            if self.last_time and state.global_step > self.last_step:
                import time
                current_time = time.time()
                steps_elapsed = state.global_step - self.last_step
                time_elapsed = current_time - self.last_time
                samples_per_sec = (steps_elapsed * args.per_device_train_batch_size * args.gradient_accumulation_steps) / time_elapsed
                log_resources(self.logger, f"Step-{state.global_step}", samples_per_sec)
                self.last_time = current_time
                self.last_step = state.global_step
            else:
                import time
                self.last_time = time.time()
                self.last_step = state.global_step
                log_resources(self.logger, f"Step-{state.global_step}")


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
    lora_rank = 2
    lora_alpha = 32
    freeze_llm = False
    freeze_vit = True
    freeze_aligner = True
    
    # Training arguments
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
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        metric_for_best_model='loss',
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=1,
        data_seed=data_seed,
        remove_unused_columns=False,
    )
    
    # Log initial resources
    log_resources(logger, "START")
    
    # Load model and setup template
    model, processor = get_model_tokenizer(model_id_or_path)
    template = get_template(model.model_meta.template, processor, default_system=None, max_length=max_length)
    template.set_mode('train')
    template.model = model
    
    # Setup LoRA
    target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                freeze_aligner=freeze_aligner)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')
    
    # Load and process dataset
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    train_processed = raw_dataset['train'].map(create_swift_format_single)
    val_processed = raw_dataset['validation'].map(create_swift_format_single)
    
    # Convert to LazyLLMDataset
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=data_seed)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=data_seed)
    
    logger.info(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)} samples")
    
    # Clear GPU cache before training
    torch.cuda.empty_cache()
    
    # Train
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        callbacks=[ResourceCallback(logger)],
    )
    
    trainer.train()
    
    # Log final resources
    log_resources(logger, "END")
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Model saved to: {trainer.state.last_model_checkpoint}')


if __name__ == "__main__":
    main()