#!/usr/bin/env python3
"""
Memory-Optimized Qwen2-VL Fine-tuning Script for Table HTML Conversion with Multi-GPU Support
NO TEDS WORKING!!!!!!!!
"""

import os
import torch
import gc
import json
from typing import List, Dict, Any
from transformers import TrainerCallback, TrainerControl, TrainerState
from PIL import Image

# Auto-detect and use all available GPUs
def setup_gpus():
    """Setup GPU configuration based on available devices"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s)")
        
        # Set memory management for better allocation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        if gpu_count > 1:
            # Use all available GPUs
            gpu_ids = ','.join(str(i) for i in range(gpu_count))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            print(f"Using GPUs: {gpu_ids}")
            return gpu_count
        else:
            # Single GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("Using single GPU: 0")
            return 1
    else:
        print("No CUDA GPUs available")
        return 0

# Setup GPUs before importing other modules
gpu_count = setup_gpus()

from swift.llm import (
    get_model_tokenizer, get_template,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset as hf_load_dataset


class InferenceCallback(TrainerCallback):
    """Custom callback to run inference on validation samples after each checkpoint save"""
    
    def __init__(self, 
                 validation_samples: List[Dict], 
                 template,
                 num_samples: int = 10,
                 output_dir: str = "inference_outputs",
                 save_images: bool = True,
                 generation_config: Dict = None):
        """
        Initialize the inference callback
        
        Args:
            validation_samples: List of validation samples in Swift format
            template: Swift template for encoding/decoding
            num_samples: Number of samples to run inference on
            output_dir: Directory to save inference results
            save_images: Whether to save images alongside results
            generation_config: Configuration for text generation
        """
        self.validation_samples = validation_samples[:num_samples]  # Take first N samples
        self.template = template
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.save_images = save_images
        self.logger = get_logger()
        
        # Default generation config
        self.generation_config = generation_config or {
            'max_new_tokens': 2048,
            'temperature': 0.1,
            'do_sample': True,
            'top_p': 0.9,
            'pad_token_id': None,  # Will be set from tokenizer
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"InferenceCallback initialized with {len(self.validation_samples)} samples")
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def save_image(self, image, step: int, sample_idx: int) -> str:
        """Save PIL image and return filename"""
        if not self.save_images:
            return None
            
        filename = f"step_{step}_sample_{sample_idx}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to PIL Image if needed
        if hasattr(image, 'save'):
            image.save(filepath)
        else:
            # Handle other image formats
            Image.fromarray(image).save(filepath)
        
        return filename
    
    def run_inference_on_sample(self, model, processor, sample: Dict, step: int, sample_idx: int) -> Dict:
        """Run inference on a single sample"""
        try:
            # Prepare the sample for inference
            # Create user message without the ground truth
            inference_sample = {
                'messages': [
                    {
                        'role': 'user',
                        'content': 'Write the HTML representation for this image of a medical table.'
                    }
                ],
                'images': sample['images']
            }
            
            # Switch template to inference mode temporarily
            original_mode = self.template.mode
            self.template.set_mode('infer')
            
            # Encode the sample
            encoded = self.template.encode(inference_sample)
            
            # Move to device
            device = next(model.parameters()).device
            input_ids = torch.tensor([encoded['input_ids']]).to(device)
            attention_mask = torch.tensor([encoded['attention_mask']]).to(device)
            
            # Handle images if present
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            
            # Add image inputs if they exist in encoded sample
            if 'pixel_values' in encoded:
                inputs['pixel_values'] = torch.tensor([encoded['pixel_values']]).to(device)
            if 'image_grid_thw' in encoded:
                inputs['image_grid_thw'] = torch.tensor([encoded['image_grid_thw']]).to(device)
            
            # Set pad_token_id if not set
            if self.generation_config['pad_token_id'] is None:
                self.generation_config['pad_token_id'] = processor.tokenizer.eos_token_id
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # Decode the generated text
            generated_text = processor.tokenizer.decode(
                outputs[0][len(input_ids[0]):], 
                skip_special_tokens=True
            )
            
            # Restore original template mode
            self.template.set_mode(original_mode)
            
            # Save image if requested
            image_filename = None
            if self.save_images and sample['images']:
                image_filename = self.save_image(sample['images'][0], step, sample_idx)
            
            # Ground truth HTML
            ground_truth = sample['messages'][1]['content']
            
            result = {
                'sample_idx': sample_idx,
                'step': step,
                'generated_html': generated_text,
                'ground_truth_html': ground_truth,
                'image_filename': image_filename,
                'input_length': len(input_ids[0]),
                'output_length': len(outputs[0]) - len(input_ids[0])
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inference failed for sample {sample_idx}: {str(e)}")
            return {
                'sample_idx': sample_idx,
                'step': step,
                'error': str(e),
                'generated_html': None,
                'ground_truth_html': sample['messages'][1]['content'] if 'messages' in sample else None
            }
    
    def run_inference(self, model, processor, step: int):
        """Run inference on all validation samples"""
        self.logger.info(f"Running inference at step {step} on {len(self.validation_samples)} samples")
        
        # Clear memory before inference
        self.clear_gpu_memory()
        
        # Switch model to eval mode
        model.eval()
        
        results = []
        
        for idx, sample in enumerate(self.validation_samples):
            self.logger.info(f"Processing sample {idx + 1}/{len(self.validation_samples)}")
            
            result = self.run_inference_on_sample(model, processor, sample, step, idx)
            results.append(result)
            
            # Clear memory after each sample if needed
            if idx % 5 == 0:  # Every 5 samples
                self.clear_gpu_memory()
        
        # Save results to file
        results_file = os.path.join(self.output_dir, f"inference_results_step_{step}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log summary
        successful_samples = [r for r in results if 'error' not in r]
        self.logger.info(f"Inference completed: {len(successful_samples)}/{len(results)} samples successful")
        
        # Switch back to training mode
        model.train()
        
        # Clear memory after inference
        self.clear_gpu_memory()
        
        return results
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Called after each checkpoint save"""
        try:
            # Get processor from template
            processor = self.template.processor
            
            # Run inference
            results = self.run_inference(model, processor, state.global_step)
            
            # Log some sample results
            if results:
                sample_result = results[0]
                if 'error' not in sample_result:
                    self.logger.info(f"Sample inference at step {state.global_step}:")
                    self.logger.info(f"Generated (first 200 chars): {sample_result['generated_html'][:200]}...")
                    self.logger.info(f"Ground truth (first 200 chars): {sample_result['ground_truth_html'][:200]}...")
            
        except Exception as e:
            self.logger.error(f"Inference callback failed at step {state.global_step}: {str(e)}")
            # Don't raise - let training continue
        
        return control


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


def filter_by_length(dataset, template, max_length):
    """Filter out samples that exceed max_length after encoding"""
    def is_valid_length(sample):
        try:
            encoded = template.encode(sample)
            return len(encoded['input_ids']) <= max_length
        except Exception:
            return False
    
    return dataset.filter(is_valid_length)


def calculate_batch_size_and_accumulation(gpu_count, base_batch_size=1, target_effective_batch_size=32):
    """Calculate optimal batch size and gradient accumulation steps for multi-GPU setup"""
    if gpu_count <= 1:
        return base_batch_size, target_effective_batch_size // base_batch_size
    
    # Scale batch size based on GPU count while being conservative about memory
    if gpu_count >= 8:
        # With 8+ GPUs, we can afford slightly larger per-device batch size
        per_device_batch_size = 2
        target_effective_batch_size = 64  # Scale up effective batch size
    elif gpu_count >= 4:
        # With 4+ GPUs, moderate scaling
        per_device_batch_size = 1
        target_effective_batch_size = 32
    else:
        # 2-3 GPUs, conservative approach
        per_device_batch_size = 1
        target_effective_batch_size = 16
    
    # Calculate gradient accumulation to maintain effective batch size
    total_batch_size_per_step = per_device_batch_size * gpu_count
    gradient_accumulation_steps = max(1, target_effective_batch_size // total_batch_size_per_step)
    
    return per_device_batch_size, gradient_accumulation_steps


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def create_inference_callback(val_processed, template, output_dir="inference_outputs"):
    """Create and return the inference callback"""
    
    # Configuration for inference
    generation_config = {
        'max_new_tokens': 2048,
        'temperature': 0.1,
        'do_sample': True,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
    }
    
    callback = InferenceCallback(
        validation_samples=val_processed,
        template=template,
        num_samples=10,  # Adjust based on your needs
        output_dir=output_dir,
        save_images=True,
        generation_config=generation_config
    )
    
    return callback


def main():
    """Main training function"""
    
    logger = get_logger()
    seed_everything(42)
    
    # Clear initial GPU memory
    clear_gpu_memory()
    
    # Configuration - Scale parameters based on GPU count
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
    output_dir = 'output'
    data_seed = 42
    
    # Scale max_length based on GPU count for better memory distribution
    if gpu_count >= 8:
        max_length = 12288  # Can handle longer sequences with more GPUs
    elif gpu_count >= 4:
        max_length = 10240  # Moderate scaling
    else:
        max_length = 8192   # Conservative for 2-3 GPUs
    
    # LoRA configuration - Scale based on GPU count
    if gpu_count >= 8:
        lora_rank = 8      # Can afford higher rank with more GPUs
        lora_alpha = 32
    elif gpu_count >= 4:
        lora_rank = 6      # Moderate scaling
        lora_alpha = 24
    else:
        lora_rank = 4      # Conservative for fewer GPUs
        lora_alpha = 16
    
    freeze_llm = False
    freeze_vit = True
    freeze_aligner = True
    
    # Calculate optimal batch size and gradient accumulation for available GPUs
    per_device_batch_size, gradient_accumulation_steps = calculate_batch_size_and_accumulation(
        gpu_count, base_batch_size=1, target_effective_batch_size=32
    )
    
    logger.info(f"Multi-GPU Scaling Configuration:")
    logger.info(f"  - GPUs available: {gpu_count}")
    logger.info(f"  - Per-device batch size: {per_device_batch_size}")
    logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"  - Effective batch size: {per_device_batch_size * gpu_count * gradient_accumulation_steps}")
    logger.info(f"  - Max sequence length: {max_length}")
    logger.info(f"  - LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    
    # Adjust workers based on GPU count
    dataloader_workers = min(4, max(2, gpu_count // 2))  # Scale workers with GPUs but cap at 4
    
    # Training arguments - Memory optimized
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
        logging_first_step=True,
        save_strategy='steps',
        save_steps=2,  # Increased to reduce I/O
        eval_strategy='steps',
        eval_steps=100,  # Increased to reduce I/O
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=3,
        metric_for_best_model='loss',
        save_total_limit=3,  # Reduced to save disk space
        logging_steps=10,  # Increased to reduce overhead
        dataloader_num_workers=dataloader_workers,  # Scale with GPU count
        data_seed=data_seed,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        # Memory optimization settings
        dataloader_pin_memory=False,  # Disable to save memory
        ddp_find_unused_parameters=False,
        ddp_timeout=1800,
        # Additional memory optimizations
        fp16=True,  # Enable mixed precision training
        dataloader_prefetch_factor=2,  # Reduce prefetch to save memory
        ddp_bucket_cap_mb=25,  # Reduce DDP bucket size
        save_safetensors=True,  # More efficient saving
    )
    
    # Load model with memory optimizations
    logger.info("Loading model with memory optimizations...")
    model, processor = get_model_tokenizer(
        model_id_or_path,
        torch_dtype=torch.float16,  # Use half precision
        device_map='auto',  # Let accelerate handle device placement
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
    )
    
    template = get_template(model.model_meta.template, processor, default_system=None, max_length=max_length)
    template.set_mode('train')
    template.model = model
    
    # Setup LoRA with memory-efficient settings
    target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                freeze_aligner=freeze_aligner)
    lora_config = LoraConfig(
        task_type='CAUSAL_LM', 
        r=lora_rank, 
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,  # Add dropout for regularization
        bias="none",  # Don't adapt bias terms to save memory
    )
    model = Swift.prepare_model(model, lora_config)
    
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')
    
    # Clear memory after model loading
    clear_gpu_memory()
    
    # Load and process dataset
    logger.info("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Apply HTML truncation to reduce sequence length
    train_processed = raw_dataset['train'].map(create_swift_format_single)
    val_processed = raw_dataset['validation'].map(create_swift_format_single)
    
    # Convert to list for callback (needed for indexing)
    val_processed_list = list(val_processed)
    
    logger.info(f"Original dataset sizes - Train: {len(train_processed)}, Val: {len(val_processed)}")
    
    # Filter out samples that are too long
    logger.info("Filtering samples by length...")
    train_processed = filter_by_length(train_processed, template, max_length)
    val_processed = filter_by_length(val_processed, template, max_length)
    
    logger.info(f"Filtered dataset sizes - Train: {len(train_processed)}, Val: {len(val_processed)}")
    
    # Convert to LazyLLMDataset
    train_dataset = LazyLLMDataset(train_processed, template.encode, random_state=data_seed)
    val_dataset = LazyLLMDataset(val_processed, template.encode, random_state=data_seed)
    
    logger.info(f"Final dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)} samples")
    
    # Check if we have enough data after filtering
    if len(train_dataset) == 0:
        logger.error("No training samples remain after filtering! Consider increasing max_length or reducing HTML truncation.")
        return
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Create inference callback
    inference_callback = create_inference_callback(
        val_processed=val_processed_list,  # Use the list version
        template=template,
        output_dir=os.path.join(output_dir, "inference_outputs")
    )
    
    # Train
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        callbacks=[inference_callback]  # Add the callback here
    )
    
    try:
        # Monitor GPU memory during training
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Clear memory on failure
        clear_gpu_memory()
        return
    
    # Save visualization
    visual_loss_dir = os.path.join(os.path.abspath(output_dir), 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)
    
    logger.info(f'Training complete. Model saved to: {trainer.state.last_model_checkpoint}')
    
    # Final memory cleanup
    clear_gpu_memory()


if __name__ == "__main__":
    main()