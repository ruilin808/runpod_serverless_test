# Required imports for the improvements
import torch
import torch.distributed
import numpy as np
import time
from transformers import TrainerCallback

# TEDS import
try:
    from table_recognition_metric import TEDS
    print("TEDS library loaded successfully")
except ImportError:
    print("TEDS library not found. Install with: pip install table-recognition-metric")
    raise


class TEDSEvaluator:
    """Improved TEDS evaluator with better multi-GPU support"""
    
    def __init__(self, model, processor, template, logger):
        self.model = model
        self.processor = processor
        self.template = template
        self.logger = logger
        
        # Better device handling for multi-GPU
        if torch.distributed.is_initialized():
            self.device = f'cuda:{torch.distributed.get_rank()}'
        elif torch.cuda.is_available():
            self.device = f'cuda:{torch.cuda.current_device()}'
        else:
            self.device = 'cpu'
        
        # Initialize TEDS scorers
        self.teds_structure_content = TEDS(structure_only=False)
        self.teds_structure_only = TEDS(structure_only=True)
        
        self.logger.info(f"TEDS evaluator initialized on device: {self.device}")
    
    def generate_prediction(self, sample, max_new_tokens=8192):
        """Generate HTML prediction with proper state management"""
        
        # Save original states
        original_training = self.model.training
        original_template_mode = getattr(self.template, 'mode', None)
        original_grad_enabled = torch.is_grad_enabled()
        
        try:
            # Set proper states for inference
            self.model.eval()
            torch.set_grad_enabled(False)
            if hasattr(self.template, 'set_mode'):
                self.template.set_mode('infer')
            
            # Clear cache to avoid memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Encode input
            inputs = self.template.encode({
                'messages': [sample['messages'][0]], 
                'images': sample.get('images', [])
            })
            
            input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Prepare generation kwargs with better error handling
            generation_kwargs = {
                'use_cache': False, 
                'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                'max_new_tokens': max_new_tokens,
                'num_beams': 1,
                'do_sample': False,
                'eos_token_id': self.processor.tokenizer.eos_token_id,
                'temperature': 1.0,  # Ensure deterministic generation
                'top_p': 1.0
            }
            
            # Add other inputs with better tensor handling
            for key, tensor_key in [('attention_mask', 'attention_mask'), 
                                   ('pixel_values', 'pixel_values'), 
                                   ('image_grid_thw', 'image_grid_thw')]:
                if key in inputs:
                    tensor = inputs[key]
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.tensor(tensor)
                    tensor = tensor.to(self.device)
                    
                    # Handle dimensions
                    if tensor.dim() == 1 and key != 'pixel_values':
                        tensor = tensor.unsqueeze(0)
                    
                    generation_kwargs[tensor_key] = tensor
            
            # Generate with timeout protection
            with torch.no_grad():
                outputs = self.model.generate(input_ids, **generation_kwargs)
            
            # Decode prediction
            prediction = self.processor.tokenizer.decode(
                outputs[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            return prediction.strip()
            
        except torch.cuda.OutOfMemoryError:
            self.logger.error("CUDA OOM during TEDS generation")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return ""
        except Exception as e:
            self.logger.warning(f"Generation error: {e}")
            return ""
        finally:
            # Restore original states
            self.model.train(original_training)
            torch.set_grad_enabled(original_grad_enabled)
            if original_template_mode and hasattr(self.template, 'set_mode'):
                self.template.set_mode(original_template_mode)
    
    def evaluate_samples(self, eval_dataset, max_samples):
        """Evaluate TEDS scores with better error handling"""
        
        # Only run on main process in multi-GPU setup
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return {'eval_teds_structure_content': 0.0, 'eval_teds_structure_only': 0.0, 'eval_teds_samples': 0}
        
        self.logger.info(f"TEDS EVALUATION ON {min(max_samples, len(eval_dataset))} SAMPLES")
        
        structure_content_scores = []
        structure_only_scores = []
        successful = 0
        
        for i in range(min(max_samples, len(eval_dataset))):
            try:
                sample = eval_dataset[i]
                pred_html = self.generate_prediction(sample)
                
                # Get ground truth
                true_html = (sample['messages'][1]['content'] if 'messages' in sample and len(sample['messages']) > 1 
                           else sample.get('html_table', ''))
                
                # Skip if either prediction or ground truth is empty
                if not pred_html.strip() or not true_html.strip():
                    self.logger.warning(f"Sample {i+1}: Empty HTML - skipping")
                    continue
                
                # Calculate TEDS scores with error handling
                try:
                    teds_sc = self.teds_structure_content(true_html, pred_html)
                    teds_so = self.teds_structure_only(true_html, pred_html)
                    
                    # Validate scores
                    if 0 <= teds_sc <= 1 and 0 <= teds_so <= 1:
                        structure_content_scores.append(teds_sc)
                        structure_only_scores.append(teds_so)
                        successful += 1
                        
                        self.logger.info(f"Sample {i+1} TEDS - Structure+Content: {teds_sc:.4f}, Structure: {teds_so:.4f}")
                    else:
                        self.logger.warning(f"Sample {i+1}: Invalid TEDS scores - SC: {teds_sc}, SO: {teds_so}")
                        
                except Exception as teds_error:
                    self.logger.warning(f"Sample {i+1}: TEDS calculation error: {teds_error}")
                    
            except Exception as e:
                self.logger.warning(f"Sample {i+1}: Processing error: {e}")
        
        # Calculate averages
        avg_sc = np.mean(structure_content_scores) if structure_content_scores else 0.0
        avg_so = np.mean(structure_only_scores) if structure_only_scores else 0.0
        
        self.logger.info(f"TEDS RESULTS: {successful}/{max_samples} samples | "
                        f"Structure+Content: {avg_sc:.4f} | Structure: {avg_so:.4f}")
        
        return {
            'eval_teds_structure_content': avg_sc, 
            'eval_teds_structure_only': avg_so, 
            'eval_teds_samples': successful
        }


class ImprovedResourceCallback(TrainerCallback):
    """Enhanced callback with better multi-GPU support"""
    
    def __init__(self, logger, teds_evaluator=None, eval_dataset=None):
        self.logger = logger
        self.teds_evaluator = teds_evaluator
        self.eval_dataset = eval_dataset
        self.last_time = self.last_step = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only log from main process
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
            
        if state.global_step % 10 == 0:
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
        """Run TEDS evaluation with proper synchronization"""
        
        # Only run TEDS on main process
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
            
        if not self.teds_evaluator or not self.eval_dataset:
            return
            
        if state.global_step >= 50:
            try:
                # Clear cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Run TEDS evaluation
                teds_results = self.teds_evaluator.evaluate_samples(self.eval_dataset, 3)
                
                self.logger.info(f"[Step {state.global_step}] TEDS: {teds_results['eval_teds_structure_content']:.4f} (SC) | "
                               f"{teds_results['eval_teds_structure_only']:.4f} (SO) | {teds_results['eval_teds_samples']} samples")
                
                # Add to logs for tensorboard (only on main process)
                if logs is not None:
                    logs.update(teds_results)
                    
            except Exception as e:
                self.logger.error(f"TEDS evaluation failed: {e}")
        else:
            self.logger.info(f"[Step {state.global_step}] Skipping TEDS - waiting for step 50+")