#!/usr/bin/env python3
"""
Optimized Qwen2-VL Fine-tuning Script with Standard LoRA
Fixed and Concise Version for 6x A40 GPUs
"""

import os
import torch
import torch.nn.functional as F
import gc
import time
import psutil
from PIL import Image
import warnings

# Auto-detect and setup GPUs
def setup_gpus():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print(f"Using {gpu_count} GPU(s)")
        return gpu_count
    else:
        print("No CUDA GPUs available")
        return 0

gpu_count = setup_gpus()

from swift.llm import get_model_tokenizer, get_template, get_multimodal_target_regex, LazyLLMDataset
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset as hf_load_dataset


def get_gpu_memory_info():
    """Get GPU memory information"""
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3
            gpu_info.append({
                'gpu_id': i, 'allocated': allocated, 'reserved': reserved,
                'total': total, 'free': total - reserved
            })
    return gpu_info


def calculate_batch_size(gpu_count, max_length=6144):
    """Calculate optimal batch size for A40 GPUs"""
    if gpu_count == 0:
        return 1, 8
    
    gpu_info = get_gpu_memory_info()
    min_free = min([info['free'] for info in gpu_info]) if gpu_info else 48.0
    
    # Memory estimate for 32B model with standard LoRA on A40
    memory_per_sample = 1.0 * (max_length / 6144)  # Scale with sequence length
    usable_memory = min_free * 0.8  # 80% utilization for A40
    
    per_device_batch = min(int(usable_memory / memory_per_sample), 3)
    per_device_batch = max(per_device_batch, 1)
    
    # Target effective batch size
    target_batch = 24 if gpu_count >= 4 else 12
    grad_accum = max(1, target_batch // (per_device_batch * gpu_count))
    
    print(f"Batch config: {per_device_batch} per device, {grad_accum} grad accum, "
          f"{per_device_batch * gpu_count * grad_accum} effective")
    
    return per_device_batch, grad_accum


def setup_optimizations():
    """Setup FlashAttention and compilation with error handling"""
    flash_enabled = False
    compile_enabled = False
    
    # Enable FlashAttention
    try:
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            flash_enabled = True
            print("✓ FlashAttention enabled")
    except Exception as e:
        print(f"⚠ FlashAttention not available: {e}")
    
    # Enable compilation
    try:
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.suppress_errors = True
            compile_enabled = True
            print("✓ Compilation optimizations enabled")
    except Exception as e:
        print(f"⚠ Compilation not available: {e}")
    
    return flash_enabled, compile_enabled


def get_lora_config(model, rank=6, alpha=24):
    """Get LoRA configuration with compatibility"""
    target_modules = get_multimodal_target_regex(
        model, freeze_llm=False, freeze_vit=True, freeze_aligner=True
    )
    
    # Standard LoRA modules
    enhanced_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
    ]
    
    all_modules = list(set(target_modules + enhanced_modules))
    
    # Build config with compatibility checks
    config_params = {
        'task_type': 'CAUSAL_LM',
        'r': rank,
        'lora_alpha': alpha,
        'target_modules': all_modules,
        'lora_dropout': 0.05,
        'bias': "none"
    }
    
    # Add newer parameters if available
    try:
        config_params['use_rslora'] = True
    except:
        pass
    
    return LoraConfig(**config_params)


def freeze_vision_layers(model):
    """Freeze early vision layers while preserving LoRA adapters"""
    frozen = 0
    
    for name, module in model.named_modules():
        if 'vision' in name and 'encoder.layers' in name:
            # Extract layer number
            try:
                layer_num = int(name.split('layers.')[1].split('.')[0])
                # Freeze first 50% of vision layers
                if hasattr(module, 'encoder') and hasattr(module.encoder, 'layers'):
                    total_layers = len(module.encoder.layers)
                    freeze_until = total_layers // 2
                    
                    if layer_num < freeze_until:
                        for param_name, param in module.named_parameters():
                            if 'lora_' not in param_name:
                                param.requires_grad = False
                                frozen += 1
            except:
                continue
    
    print(f"Frozen {frozen} vision parameters")
    return frozen


def preprocess_sample(sample):
    """Efficient sample preprocessing"""
    image = sample['image']
    
    # KEEP Commented out: Resize image if needed (compatible with older PIL)
    # if hasattr(image, 'size') and max(image.size) > 1280:
    #    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    
    return {
        'messages': [
            {'role': 'user', 'content': 'Write the HTML representation for this image of a medical table.'},
            {'role': 'assistant', 'content': sample['html_table']}
        ],
        'images': [image]
    }


def filter_by_length(dataset, template, max_length):
    """Filter samples by sequence length"""
    def is_valid(sample):
        try:
            encoded = template.encode(sample)
            return len(encoded['input_ids']) <= max_length
        except:
            return False
    
    return dataset.filter(is_valid)


class SimpleCallback:
    """Simple training callback for monitoring"""
    def __init__(self):
        self.start_time = time.time()
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'train_loss' in logs:
            self.step_count += 1
            if state.global_step % 10 == 0:
                elapsed = time.time() - self.start_time
                print(f"Step {state.global_step}: Loss={logs['train_loss']:.4f}, "
                      f"Time={elapsed/60:.1f}min")


def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def main():
    """Main training function"""
    logger = get_logger()
    seed_everything(42)
    
    print("=== Qwen2-VL Fine-tuning with Standard LoRA ===")
    
    # Setup optimizations
    flash_enabled, compile_enabled = setup_optimizations()
    clear_memory()
    
    # Configuration
    model_path = 'Qwen/Qwen2.5-VL-32B-Instruct'
    output_dir = 'output'
    max_length = 6144  # Conservative for A40 with 32B model
    lora_rank = 6
    lora_alpha = 24
    
    # Calculate batch configuration
    per_device_batch, grad_accum = calculate_batch_size(gpu_count, max_length)
    
    print(f"Config: max_length={max_length}, rank={lora_rank}, "
          f"batch={per_device_batch}, accum={grad_accum}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=10,
        
        # Memory optimizations
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        torch_empty_cache_steps=20,
        
        # Optimizer
        optim="adamw_torch",
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        
        # Logging and saving
        logging_steps=10,
        save_steps=10,
        eval_steps=10,
        save_strategy='steps',
        eval_strategy='steps',
        save_total_limit=2,
        
        # Multi-GPU
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=25,
        
        # Optional compilation
        torch_compile=compile_enabled,
        torch_compile_backend="inductor" if compile_enabled else None,
        
        # Other
        remove_unused_columns=False,
        report_to=['tensorboard'],
        prediction_loss_only=True,
    )
    
    # Load model
    print("Loading model...")
    try:
        model, processor = get_model_tokenizer(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Setup template
    try:
        template = get_template(model.model_meta.template, processor, 
                              default_system=None, max_length=max_length)
    except AttributeError:
        # Fallback for different model structures
        template = get_template('qwen-vl', processor, 
                              default_system=None, max_length=max_length)
    
    template.set_mode('train')
    
    # Apply LoRA
    print("Applying LoRA...")
    lora_config = get_lora_config(model, lora_rank, lora_alpha)
    model = Swift.prepare_model(model, lora_config)
    
    # Print model info
    model_info = get_model_parameter_info(model)
    print(f"Model info: {model_info}")
    
    # Freeze vision layers (after LoRA)
    freeze_vision_layers(model)
    
    # Memory check
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        max_used = max([info['reserved'] for info in gpu_info])
        print(f"GPU memory after model loading: {max_used:.1f}GB")
    
    # Load dataset
    print("Loading dataset...")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    
    # Process dataset
    train_data = raw_dataset['train'].map(preprocess_sample)
    val_data = raw_dataset['validation'].map(preprocess_sample)
    
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val samples")
    
    # Filter by length
    train_data = filter_by_length(train_data, template, max_length)
    val_data = filter_by_length(val_data, template, max_length)
    
    print(f"After filtering: {len(train_data)} train, {len(val_data)} val samples")
    
    if len(train_data) == 0:
        logger.error("No samples after filtering! Increase max_length.")
        return
    
    # Create datasets
    train_dataset = LazyLLMDataset(train_data, template.encode, random_state=42)
    val_dataset = LazyLLMDataset(val_data, template.encode, random_state=42)
    
    # Setup trainer
    callback = SimpleCallback()
    
    model.enable_input_require_grads()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
        callbacks=[callback],
    )
    
    # Train
    print(f"Starting training: {len(train_dataset)} samples, "
          f"{len(train_dataset) // (per_device_batch * gpu_count * grad_accum)} steps/epoch")
    
    try:
        start_time = time.time()
        trainer.train()
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n=== Training Complete ===")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Steps completed: {callback.step_count}")
        print(f"Model saved to: {trainer.state.last_model_checkpoint}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        clear_memory()
        raise
    
    # Save visualizations
    try:
        visual_dir = os.path.join(output_dir, 'visual_loss')
        os.makedirs(visual_dir, exist_ok=True)
        plot_images(visual_dir, training_args.logging_dir, ['train/loss'], 0.9)
        print("✓ Visualizations saved")
    except Exception as e:
        print(f"⚠ Could not save visualizations: {e}")
    
    # Final cleanup
    clear_memory()
    print("✓ Training completed successfully")


if __name__ == "__main__":
    main()