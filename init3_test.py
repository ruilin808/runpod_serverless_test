#!/usr/bin/env python3
"""
Debug version of Qwen2-VL Fine-tuning Script
"""

import os
import sys
import traceback

print("=== SCRIPT STARTING ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    print("Step 1: Importing basic libraries...")
    import psutil
    import torch
    import time
    import numpy as np
    print("✓ Basic libraries imported successfully")

    print("Step 2: Checking CUDA...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
        print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA GPUs available")

    print("Step 3: Importing Swift libraries...")
    try:
        from swift.llm import get_model_tokenizer, get_template, get_multimodal_target_regex, LazyLLMDataset
        print("✓ swift.llm imported")
    except ImportError as e:
        print(f"✗ Error importing swift.llm: {e}")
        raise

    try:
        from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
        print("✓ swift.utils imported")
    except ImportError as e:
        print(f"✗ Error importing swift.utils: {e}")
        raise

    try:
        from swift.tuners import Swift, LoraConfig
        print("✓ swift.tuners imported")
    except ImportError as e:
        print(f"✗ Error importing swift.tuners: {e}")
        raise

    try:
        from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
        print("✓ swift.trainers imported")
    except ImportError as e:
        print(f"✗ Error importing swift.trainers: {e}")
        raise

    print("Step 4: Importing other libraries...")
    try:
        from datasets import load_dataset as hf_load_dataset
        print("✓ datasets imported")
    except ImportError as e:
        print(f"✗ Error importing datasets: {e}")
        raise

    try:
        from transformers import TrainerCallback
        print("✓ transformers.TrainerCallback imported")
    except ImportError as e:
        print(f"✗ Error importing TrainerCallback: {e}")
        raise

    print("Step 5: Importing TEDS...")
    try:
        from table_recognition_metric import TEDS
        print("✓ TEDS library loaded successfully")
    except ImportError as e:
        print(f"✗ TEDS library not found: {e}")
        print("Install with: pip install table-recognition-metric")
        raise

    print("Step 6: Defining functions...")
    
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
        try:
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
        except Exception as e:
            print(f"Error in log_resources: {e}")

    print("✓ Functions defined")

    print("Step 7: Defining TEDSEvaluator class...")
    
    class TEDSEvaluator:
        """TEDS evaluator for table structure recognition"""
        
        def __init__(self, model, processor, template, logger):
            print("Initializing TEDSEvaluator...")
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

    print("✓ TEDSEvaluator class defined")

    print("Step 8: Defining ResourceCallback class...")
    
    class ResourceCallback(TrainerCallback):
        """Enhanced callback for resource logging and TEDS evaluation"""
        def __init__(self, logger, teds_evaluator=None, eval_dataset=None):
            print("Initializing ResourceCallback...")
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

    print("✓ ResourceCallback class defined")

    print("Step 9: Defining main function...")
    
    def main():
        """Main training function"""
        print("=== ENTERING MAIN FUNCTION ===")
        
        print("Getting logger...")
        logger = get_logger()
        print("✓ Logger obtained")
        
        print("Setting seed...")
        seed_everything(42)
        print("✓ Seed set")
        
        # Configuration
        model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
        output_dir = 'output'
        
        print(f"Model path: {model_id_or_path}")
        print(f"Output dir: {output_dir}")
        
        # Calculate training parameters
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        gradient_accumulation_steps = max(4 // gpu_count, 1)
        total_samples, epochs = 222, 6
        expected_steps = (total_samples * epochs) // (1 * gradient_accumulation_steps * gpu_count)
        
        logger.info(f"Training setup: {gpu_count} GPUs, grad_accum={gradient_accumulation_steps}, expected_steps={expected_steps}")
        
        print("Creating training arguments...")
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
        print("✓ Training arguments created")
        
        log_resources(logger, "START")
        
        # Load model and setup
        print("Loading model...")
        logger.info("Loading model...")
        try:
            model, processor = get_model_tokenizer(model_id_or_path)
            print("✓ Model and processor loaded")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            traceback.print_exc()
            raise
        
        print("Getting template...")
        try:
            template = get_template(model.model_meta.template, processor, max_length=32768)
            template.set_mode('train')
            if template.use_model:
                template.model = model
            print("✓ Template configured")
        except Exception as e:
            print(f"✗ Error with template: {e}")
            traceback.print_exc()
            raise
        
        # Setup LoRA
        print("Setting up LoRA...")
        logger.info("Setting up LoRA...")
        try:
            target_modules = get_multimodal_target_regex(model, freeze_llm=False, freeze_vit=True, freeze_aligner=True)
            lora_config = LoraConfig(task_type='CAUSAL_LM', r=4, lora_alpha=16, target_modules=target_modules)
            model = Swift.prepare_model(model, lora_config)
            print("✓ LoRA configured")
        except Exception as e:
            print(f"✗ Error setting up LoRA: {e}")
            traceback.print_exc()
            raise
        
        logger.info(f'Model: {get_model_parameter_info(model)}')
        
        # Load and process dataset
        print("Loading dataset...")
        logger.info("Loading dataset...")
        try:
            raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
            print("✓ Raw dataset loaded")
            
            train_processed = raw_dataset['train'].map(create_swift_format_single)
            val_processed = raw_dataset['validation'].map(create_swift_format_single)
            print("✓ Dataset processed")
            
            train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=42)
            val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=42)
            print("✓ Lazy datasets created")
            
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            traceback.print_exc()
            raise
        
        logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        
        # Initialize TEDS evaluator
        print("Initializing TEDS evaluator...")
        try:
            teds_evaluator = TEDSEvaluator(model, processor, template, logger)
            print("✓ TEDS evaluator initialized")
        except Exception as e:
            print(f"✗ Error initializing TEDS: {e}")
            traceback.print_exc()
            raise
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        print("Starting training...")
        logger.info("Starting training...")
        try:
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
            print("✓ Trainer created")
            
            trainer.train()
            print("✓ Training completed")
            
        except Exception as e:
            print(f"✗ Error during training: {e}")
            traceback.print_exc()
            raise
        
        # Final evaluation
        print("Final TEDS evaluation...")
        logger.info("Final TEDS evaluation...")
        try:
            final_results = teds_evaluator.evaluate_samples(val_processed, 10)
            print("✓ Final evaluation completed")
        except Exception as e:
            print(f"✗ Error in final evaluation: {e}")
            traceback.print_exc()
        
        log_resources(logger, "END")
        
        # Save visualization
        print("Saving visualization...")
        try:
            visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
            os.makedirs(visual_loss_dir, exist_ok=True)
            plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
            print("✓ Visualization saved")
        except Exception as e:
            print(f"✗ Error saving visualization: {e}")
            traceback.print_exc()
        
        logger.info(f'Training complete. Best model: {trainer.state.best_model_checkpoint}')
        logger.info(f'Final TEDS - Structure+Content: {final_results["eval_teds_structure_content"]:.4f} | '
                    f'Structure: {final_results["eval_teds_structure_only"]:.4f}')
        
        print("=== MAIN FUNCTION COMPLETED ===")

    print("✓ Main function defined")

    print("Step 10: Starting main execution...")
    if __name__ == "__main__":
        print("=== CALLING MAIN ===")
        main()

except Exception as e:
    print(f"FATAL ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

print("=== SCRIPT COMPLETED SUCCESSFULLY ===")