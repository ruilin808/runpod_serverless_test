#!/usr/bin/env python3
"""
TEDS Evaluation Integration for Table HTML Conversion Training
"""

import random
import torch
from table_recognition_metric import TEDS
from swift.utils import get_logger

def setup_teds_evaluation():
    """Initialize TEDS metric instances"""
    return {
        'teds_full': TEDS(structure_only=False),  # Full evaluation (structure + content)
        'teds_structure': TEDS(structure_only=True)  # Structure-only evaluation
    }

def run_inference_on_sample(model, template, sample, max_new_tokens=2048):
    """
    Run inference on a single sample to generate HTML prediction
    
    Args:
        model: The trained model
        template: The template for encoding/decoding
        sample: Single sample from validation dataset
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        str: Generated HTML string
    """
    try:
        # Prepare the input message (same format as training)
        messages = [
            {
                'role': 'user',
                'content': 'Write the HTML representation for this image of a medical table.'
            }
        ]
        
        # Create input with image
        input_data = {
            'messages': messages,
            'images': [sample['image']]
        }
        
        # Encode input
        encoded = template.encode(input_data)
        
        # Move to device
        input_ids = encoded['input_ids'].unsqueeze(0).to(model.device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(model.device)
        
        # Handle images if present
        if 'images' in encoded:
            images = encoded['images']
            if isinstance(images, list):
                images = [img.to(model.device) if hasattr(img, 'to') else img for img in images]
            else:
                images = images.to(model.device)
        else:
            images = None
        
        # Generate response
        model.eval()
        with torch.no_grad():
            if images is not None:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=template.tokenizer.eos_token_id
                )
            else:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=template.tokenizer.eos_token_id
                )
        
        # Decode response (only the new tokens)
        new_tokens = generated_ids[0][input_ids.shape[1]:]
        predicted_html = template.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return predicted_html.strip()
        
    except Exception as e:
        logger = get_logger()
        logger.error(f"Inference failed for sample: {e}")
        return ""

def evaluate_teds_on_validation(model, template, val_dataset, teds_metrics, 
                               num_samples=3, max_new_tokens=2048):
    """
    Evaluate TEDS scores on a subset of validation samples
    
    Args:
        model: The trained model
        template: The template for encoding/decoding
        val_dataset: Validation dataset
        teds_metrics: Dictionary containing TEDS metric instances
        num_samples: Number of samples to evaluate
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        dict: Dictionary containing evaluation results
    """
    logger = get_logger()
    
    # Randomly sample validation examples
    sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    
    teds_full_scores = []
    teds_structure_scores = []
    
    logger.info(f"Evaluating TEDS on {len(sample_indices)} validation samples...")
    
    for i, idx in enumerate(sample_indices):
        try:
            # Get the original sample (before Swift format conversion)
            sample = val_dataset.dataset[idx]  # Access original dataset
            
            # Run inference
            predicted_html = run_inference_on_sample(model, template, sample, max_new_tokens)
            
            # Get ground truth HTML
            gt_html = sample['html_table']
            
            # Calculate TEDS scores
            if predicted_html and gt_html:
                # Full TEDS (structure + content)
                teds_full_score = teds_metrics['teds_full'](gt_html, predicted_html)
                teds_full_scores.append(teds_full_score)
                
                # Structure-only TEDS
                teds_structure_score = teds_metrics['teds_structure'](gt_html, predicted_html)
                teds_structure_scores.append(teds_structure_score)
                
                logger.info(f"Sample {i+1}/{len(sample_indices)}: "
                           f"TEDS_full={teds_full_score:.4f}, "
                           f"TEDS_structure={teds_structure_score:.4f}")
            else:
                logger.warning(f"Sample {i+1}/{len(sample_indices)}: Empty prediction or ground truth")
                
        except Exception as e:
            logger.error(f"Error evaluating sample {i+1}/{len(sample_indices)}: {e}")
            continue
    
    # Calculate average scores
    results = {
        'teds_full_avg': sum(teds_full_scores) / len(teds_full_scores) if teds_full_scores else 0.0,
        'teds_structure_avg': sum(teds_structure_scores) / len(teds_structure_scores) if teds_structure_scores else 0.0,
        'num_evaluated': len(teds_full_scores),
        'teds_full_scores': teds_full_scores,
        'teds_structure_scores': teds_structure_scores
    }
    
    logger.info(f"TEDS Evaluation Results:")
    logger.info(f"  - Samples evaluated: {results['num_evaluated']}")
    logger.info(f"  - Average TEDS (full): {results['teds_full_avg']:.4f}")
    logger.info(f"  - Average TEDS (structure): {results['teds_structure_avg']:.4f}")
    
    return results

# Custom callback class for TEDS evaluation
class TEDSEvaluationCallback:
    """Custom callback to run TEDS evaluation during training"""
    
    def __init__(self, model, template, val_dataset, eval_steps=100, num_samples=3):
        self.model = model
        self.template = template
        self.val_dataset = val_dataset
        self.eval_steps = eval_steps
        self.num_samples = num_samples
        self.teds_metrics = setup_teds_evaluation()
        self.step_count = 0
        
    def on_step_end(self, trainer, step):
        """Called at the end of each training step"""
        self.step_count += 1
        
        if self.step_count % self.eval_steps == 0:
            logger = get_logger()
            logger.info(f"Running TEDS evaluation at step {self.step_count}")
            
            # Run TEDS evaluation
            results = evaluate_teds_on_validation(
                self.model, 
                self.template, 
                self.val_dataset, 
                self.teds_metrics,
                num_samples=self.num_samples
            )
            
            # Log results to tensorboard if available
            if hasattr(trainer, 'log'):
                trainer.log({
                    'eval/teds_full': results['teds_full_avg'],
                    'eval/teds_structure': results['teds_structure_avg'],
                    'eval/teds_samples_evaluated': results['num_evaluated']
                })

# Integration function to add to your main training script
def integrate_teds_evaluation_into_training(trainer, model, template, val_dataset, 
                                          eval_steps=100, num_samples=3):
    """
    Integrate TEDS evaluation into the training process
    
    Args:
        trainer: The Seq2SeqTrainer instance
        model: The model being trained
        template: The template for encoding/decoding
        val_dataset: Validation dataset
        eval_steps: How often to run TEDS evaluation
        num_samples: Number of samples to evaluate each time
    """
    
    # Create callback
    teds_callback = TEDSEvaluationCallback(
        model=model,
        template=template,
        val_dataset=val_dataset,
        eval_steps=eval_steps,
        num_samples=num_samples
    )
    
    # Add callback to trainer
    trainer.add_callback(teds_callback)
    
    return trainer

# Standalone evaluation function for after training
def evaluate_model_with_teds(model, template, val_dataset, num_samples=10):
    """
    Standalone function to evaluate trained model with TEDS
    
    Args:
        model: Trained model
        template: Template for encoding/decoding
        val_dataset: Validation dataset
        num_samples: Number of samples to evaluate
    """
    
    teds_metrics = setup_teds_evaluation()
    
    results = evaluate_teds_on_validation(
        model=model,
        template=template, 
        val_dataset=val_dataset,
        teds_metrics=teds_metrics,
        num_samples=num_samples
    )
    
    return results