#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script for Table HTML Conversion with TEDS Evaluation
"""

import os
import psutil
import torch
import time
import numpy as np

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


class TEDSEvaluator:
    """TEDS evaluator for table structure recognition"""
    
    def __init__(self, model, processor, template, logger):
        self.model = model
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
            
            # Get current device from model
            device = next(self.model.parameters()).device
            
            with torch.no_grad():
                # Properly format the input for generation
                messages = [sample['messages'][0]]  # User message only
                images = sample.get('images', [])
                
                # Encode input with proper template
                inputs = self.template.encode({
                    'messages': messages, 
                    'images': images
                })
                
                # Move tensors to correct device
                input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                
                # Prepare generation kwargs
                generation_kwargs = {
                    'use_cache': False,
                    'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                    'max_new_tokens': max_new_tokens,
                    'num_beams': 1,
                    'do_sample': False,
                    'eos_token_id': self.processor.tokenizer.eos_token_id,
                    'temperature': 1.0,
                    'top_p': 1.0
                }
                
                # Add other inputs with proper device handling
                for key, tensor_key in [
                    ('attention_mask', 'attention_mask'), 
                    ('pixel_values', 'pixel_values'), 
                    ('image_grid_thw', 'image_grid_thw')
                ]:
                    if key in inputs and inputs[key] is not None:
                        tensor = inputs[key]
                        if not isinstance(tensor, torch.Tensor):
                            tensor = torch.tensor(tensor)
                        
                        tensor = tensor.to(device)
                        
                        # Ensure proper batch dimension
                        if tensor.dim() == 1 and key != 'pixel_values':
                            tensor = tensor.unsqueeze(0)
                        elif key == 'pixel_values' and tensor.dim() == 4:
                            # pixel_values might need proper batching
                            pass
                        
                        generation_kwargs[tensor_key] = tensor
                
                # Generate
                with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for generation
                    outputs = self.model.generate(input_ids, **generation_kwargs)
                
                # Decode only the generated part
                generated_tokens = outputs[0][input_ids.shape[1]:]
                prediction = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                return prediction.strip()
                
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"CUDA OOM during generation: {e}")
            torch.cuda.empty_cache()
            return ""
        except Exception as e:
            self.logger.error(f"Generation error: {type(e).__name__}: {e}")
            return ""
        finally:
            # Always cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def evaluate_samples(self, eval_dataset, max_samples):
        """Evaluate TEDS scores on a subset of samples"""
        self.logger.info(f"TEDS EVALUATION ON {min(max_samples, len(eval_dataset))} SAMPLES")
        
        structure_content_scores = []
        structure_only_scores = []
        successful = 0
        
        # Set model to eval mode
        original_mode = self.model.training
        self.model.eval()
        
        try:
            for i in range(min(max_samples, len(eval_dataset))):
                try:
                    sample = eval_dataset[i]
                    
                    # Generate prediction
                    pred_html = self.generate_prediction(sample)
                    
                    # Get ground truth
                    if 'messages' in sample and len(sample['messages']) > 1:
                        true_html = sample['messages'][1]['content']
                    else:
                        true_html = sample.get('html_table', '')
                    
                    # Validate HTML content
                    if not pred_html.strip():
                        self.logger.warning(f"Sample {i+1}: Empty prediction")
                        continue
                    
                    if not true_html.strip():
                        self.logger.warning(f"Sample {i+1}: Empty ground truth")
                        continue
                    
                    # Calculate TEDS scores with error handling
                    try:
                        teds_sc = self.teds_structure_content(true_html, pred_html)
                        teds_so = self.teds_structure_only(true_html, pred_html)
                        
                        # Validate scores
                        if not (0.0 <= teds_sc <= 1.0) or not (0.0 <= teds_so <= 1.0):
                            self.logger.warning(f"Sample {i+1}: Invalid TEDS scores - SC: {teds_sc}, SO: {teds_so}")
                            continue
                        
                        structure_content_scores.append(teds_sc)
                        structure_only_scores.append(teds_so)
                        successful += 1
                        
                        self.logger.info(f"Sample {i+1} TEDS - Structure+Content: {teds_sc:.4f}, Structure: {teds_so:.4f}")
                        
                    except Exception as teds_error:
                        self.logger.warning(f"Sample {i+1}: TEDS calculation error: {teds_error}")
                        continue
                
                except Exception as e:
                    self.logger.warning(f"Sample {i+1}: Processing error: {e}")
                    continue
            
            # Calculate averages
            avg_sc = np.mean(structure_content_scores) if structure_content_scores else 0.0
            avg_so = np.mean(structure_only_scores) if structure_only_scores else 0.0
            
            self.logger.info(f"TEDS RESULTS: {successful}/{max_samples} samples evaluated successfully")
            self.logger.info(f"Average TEDS - Structure+Content: {avg_sc:.4f} | Structure-only: {avg_so:.4f}")
            
            return {
                'eval_teds_structure_content': avg_sc,
                'eval_teds_structure_only': avg_so,
                'eval_teds_samples': successful
            }
            
        finally:
            # Restore original training mode
            self.model.train(original_mode)
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
            
        # Only evaluate after sufficient training
        if state.global_step >= 50:
            try:
                self.logger.info(f"Running TEDS evaluation at step {state.global_step}")
                
                # Run TEDS evaluation
                teds_results = self.teds_evaluator.evaluate_samples(self.eval_dataset, 5)  # Increased to 5 samples
                
                # Log results
                self.logger.info(f"[Step {state.global_step}] TEDS Results:")
                self.logger.info(f"  Structure+Content: {teds_results['eval_teds_structure_content']:.4f}")
                self.logger.info(f"  Structure-only: {teds_results['eval_teds_structure_only']:.4f}")
                self.logger.info(f"  Successful samples: {teds_results['eval_teds_samples']}")
                
                # Add to logs for tensorboard
                if logs is not None:
                    logs.update(teds_results)
                    
            except Exception as e:
                self.logger.error(f"TEDS evaluation failed at step {state.global_step}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
        else:
            self.logger.info(f"[Step {state.global_step}] Skipping TEDS evaluation - waiting for step 50+")


def main():
    """Main training function"""
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
    
    logger.info(f"Training setup: {gpu_count} GPUs, grad_accum={gradient_accumulation_steps}, expected_steps={expected_steps}")
    
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
    raw_dataset = hf_load_dataset("apoidea/pubtabnet-html") # ruilin808/dataset_1920x1280
    train_processed = raw_dataset['train'].map(create_swift_format_single)
    val_processed = raw_dataset['validation'].map(create_swift_format_single)
    
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
        callbacks=[ResourceCallback(logger, teds_evaluator, val_processed)],
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