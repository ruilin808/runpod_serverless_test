import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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
    
    def generate_prediction(self, sample, max_new_tokens=4096):  # Reduced max tokens
        """Generate HTML prediction for a sample"""
        try:
            self.logger.info(f"Starting generation with max_new_tokens={max_new_tokens}")
            self.model.eval()
            
            with torch.no_grad():
                # Log input processing
                self.logger.info("Encoding input...")
                inputs = self.template.encode({'messages': [sample['messages'][0]], 'images': sample.get('images', [])})
                
                self.logger.info(f"Input shape: {len(inputs['input_ids'])}")
                input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long, device=self.device).unsqueeze(0)
                
                # Clean generation kwargs - remove invalid parameters
                generation_kwargs = {
                    'max_new_tokens': max_new_tokens,
                    'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                    'eos_token_id': self.processor.tokenizer.eos_token_id,
                    'do_sample': False,
                    'num_beams': 1,
                    'use_cache': False,
                    # Remove temperature, top_p, top_k - these cause the warning
                }
                
                # Add only essential tensors
                if 'attention_mask' in inputs:
                    tensor = torch.tensor(inputs['attention_mask']).to(self.device).unsqueeze(0)
                    generation_kwargs['attention_mask'] = tensor
                    self.logger.info(f"Added attention_mask: {tensor.shape}")
                
                if 'pixel_values' in inputs:
                    tensor = torch.tensor(inputs['pixel_values']).to(self.device)
                    generation_kwargs['pixel_values'] = tensor
                    self.logger.info(f"Added pixel_values: {tensor.shape}")
                
                if 'image_grid_thw' in inputs:
                    tensor = torch.tensor(inputs['image_grid_thw']).to(self.device)
                    generation_kwargs['image_grid_thw'] = tensor
                    self.logger.info(f"Added image_grid_thw: {tensor.shape}")
                
                self.logger.info("Starting model.generate()...")
                outputs = self.model.generate(input_ids, **generation_kwargs)
                
                self.logger.info("Generation completed, decoding...")
                prediction = self.processor.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                
                self.logger.info(f"Prediction length: {len(prediction)}")
                return prediction.strip()
                
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return ""
    
    def evaluate_samples(self, eval_dataset, max_samples):
        """Evaluate TEDS scores on a subset of samples"""
        self.logger.info(f"TEDS EVALUATION ON {min(max_samples, len(eval_dataset))} SAMPLES")
        
        structure_content_scores, structure_only_scores, successful = [], [], 0
        
        for i in range(min(max_samples, len(eval_dataset))):
            try:
                self.logger.info(f"Processing sample {i+1}/{min(max_samples, len(eval_dataset))}")
                sample = eval_dataset[i]
                
                # Add timeout protection
                with timeout(60):  # 60 second timeout per sample
                    pred_html = self.generate_prediction(sample, max_new_tokens=4096)  # Reduced max tokens
                
                # Get ground truth
                true_html = (sample['messages'][1]['content'] if 'messages' in sample and len(sample['messages']) > 1 
                           else sample.get('html_table', ''))
                
                if not pred_html.strip() or not true_html.strip():
                    self.logger.warning(f"Sample {i+1}: Empty HTML (pred: {len(pred_html)}, true: {len(true_html)})")
                    continue
                
                # Calculate TEDS scores
                self.logger.info(f"Sample {i+1}: Calculating TEDS scores...")
                teds_sc = self.teds_structure_content(true_html, pred_html)
                teds_so = self.teds_structure_only(true_html, pred_html)
                
                structure_content_scores.append(teds_sc)
                structure_only_scores.append(teds_so)
                successful += 1
                
                self.logger.info(f"Sample {i+1} TEDS - Structure+Content: {teds_sc:.4f}, Structure: {teds_so:.4f}")
                
            except TimeoutError:
                self.logger.warning(f"Sample {i+1}: Generation timed out after 60 seconds")
                continue
            except Exception as e:
                self.logger.warning(f"Sample {i+1}: Error: {e}")
                continue
        
        # Calculate averages
        avg_sc = np.mean(structure_content_scores) if structure_content_scores else 0.0
        avg_so = np.mean(structure_only_scores) if structure_only_scores else 0.0
        
        self.logger.info(f"TEDS RESULTS: {successful}/{max_samples} samples | Structure+Content: {avg_sc:.4f} | Structure: {avg_so:.4f}")
        
        return {'eval_teds_structure_content': avg_sc, 'eval_teds_structure_only': avg_so, 'eval_teds_samples': successful}