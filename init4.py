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