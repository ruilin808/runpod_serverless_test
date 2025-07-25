## Optimized ruilin808/dataset Training - CUDA Multi-GPU

# Cell 1: Enhanced CUDA multi-GPU optimization
import os
import gc
import psutil
import torch
import numpy as np
from datasets import load_dataset as hf_load_dataset
import multiprocessing as mp
import threading
import time

# Enhanced CUDA multi-GPU configuration
print("Configuring for CUDA multi-GPU mode...")

# Remove macOS-specific environment variables
if 'PYTORCH_ENABLE_MPS_FALLBACK' in os.environ:
    del os.environ['PYTORCH_ENABLE_MPS_FALLBACK']
if 'PYTORCH_MPS_HIGH_WATERMARK_RATIO' in os.environ:
    del os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']

# CUDA optimization settings
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Allow async kernel launches
os.environ['CUDA_CACHE_DISABLE'] = '0'     # Enable CUDA cache
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings

# Multi-threading optimization
total_cores = psutil.cpu_count(logical=True)
physical_cores = psutil.cpu_count(logical=False)
optimal_cores = min(16, physical_cores)

os.environ['OMP_NUM_THREADS'] = str(optimal_cores)
os.environ['MKL_NUM_THREADS'] = str(optimal_cores)

# Check GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a CUDA-compatible GPU and PyTorch with CUDA support.")

num_gpus = torch.cuda.device_count()
print(f"✓ Found {num_gpus} CUDA GPU(s)")

for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

# Set default device and tensor type
torch.set_default_device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_num_threads(optimal_cores)

print(f"✓ Using {optimal_cores} CPU cores for data loading")

# Enhanced memory management
def log_memory(stage=""):
    """Enhanced memory logging with GPU info"""
    process = psutil.Process()
    cpu_memory_gb = process.memory_info().rss / 1024**3
    virtual_memory = psutil.virtual_memory()
    available_gb = virtual_memory.available / 1024**3
    used_percent = virtual_memory.percent
    
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_info.append(f"GPU{i}: {allocated:.1f}/{total:.1f}GB (cached: {cached:.1f}GB)")
    
    gpu_str = " | ".join(gpu_info) if gpu_info else "No GPU"
    print(f"[MEMORY {stage}] CPU: {cpu_memory_gb:.1f}GB ({used_percent:.1f}% used, {available_gb:.1f}GB available) | {gpu_str}")

def aggressive_cleanup():
    """Enhanced cleanup for multi-GPU setup"""
    # Clear Python objects
    for _ in range(3):
        gc.collect()
    
    # Clear GPU memory on all devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    # Force garbage collection of tensors
    import weakref
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            try:
                del obj
            except:
                pass
    
    gc.collect()

log_memory("Initial")

from swift.llm import (
    get_model_tokenizer, get_template, LazyLLMDataset,
    get_multimodal_target_regex
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments

log_memory("After imports")

# TEDS import
try:
    from table_recognition_metric import TEDS
    print("✓ TEDS library loaded")
except ImportError:
    print("❌ TEDS library not found. Install with: pip install table-recognition-metric")
    raise

logger = get_logger()
seed_everything(42)

# Configuration variables
model_id_or_path = 'Qwen/Qwen2.5-VL-32B-Instruct'
output_dir = './output'
learning_rate = 2e-5
weight_decay = 0.01
lr_scheduler_type = 'cosine'
warmup_ratio = 0.1
report_to = ['tensorboard']
logging_first_step = True
num_train_epochs = 2
metric_for_best_model = 'eval_loss'
greater_is_better = False
load_best_model_at_end = True
max_grad_norm = 1.0
system = None

# TOKEN LIMITS CONFIGURATION
max_length = 32768              # Maximum input tokens (text + image tokens combined)
max_new_tokens_train = 16384     # Maximum output tokens during training  
max_new_tokens_eval = 16384     # Maximum output tokens during TEDS evaluation
max_new_tokens_inference = 16384 # Maximum output tokens for general inference

data_seed = 42
lora_rank = 8
lora_alpha = 16
freeze_llm = False
freeze_vit = True
freeze_aligner = True

print(f"TOKEN LIMITS:")
print(f"  Input (text + image): {max_length:,} tokens")
print(f"  Output (training): {max_new_tokens_train:,} tokens") 
print(f"  Output (TEDS eval): {max_new_tokens_eval:,} tokens")
print(f"  Output (inference): {max_new_tokens_inference:,} tokens")

# Enhanced vision processing parameters for GPU
os.environ['MAX_PIXELS'] = '6422528'   # ~6.4M pixels max per image
os.environ['MIN_PIXELS'] = '3136'      # ~3K pixels min per image  
os.environ['IMAGE_FACTOR'] = '28'      # Image patch size factor

# IMAGE TOKEN CALCULATION
# Qwen2.5-VL uses dynamic image tokenization based on image size
# Formula: image_tokens ≈ (height // 14) * (width // 14) * patches_per_tile
# For MAX_PIXELS=6422528: roughly 2800x2300 → ~200*164 → ~32,800 image tokens
# This means images can use up to ~33K of the 32K input token limit

print(f"IMAGE PROCESSING:")
print(f"  Max pixels per image: {int(os.environ['MAX_PIXELS']):,}")
print(f"  Estimated max image tokens: ~33,000")
print(f"  Remaining tokens for text: ~{max_length - 33000:,}")

# Cell 2: Optimized format conversion with batching
def create_ms_swift_format(batch):
    """Convert batch of samples to ms-swift format - OPTIMIZED FOR CLINICAL TABLES"""
    return {
        'messages': [
            [
                {
                    'role': 'user',
                    'content': '<image>\nConvert this clinical table to HTML format. Preserve all structure, rowspan, colspan, and medical terminology exactly.'
                },
                {
                    'role': 'assistant',
                    'content': html_table
                }
            ]
            for html_table in batch['html_table']
        ],
        'images': [[image] for image in batch['image']]
    }

def create_ms_swift_format_single(sample):
    """Convert custom dataset sample to ms-swift format"""
    return {
        'messages': [
            {
                'role': 'user',
                'content': '<image>\nPlease convert this table image to HTML format.'
            },
            {
                'role': 'assistant',
                'content': sample['html_table']
            }
        ],
        'images': [sample['image']]
    }

# Cell 3: CUDA-optimized TEDS Evaluator
class CUDATEDSEvaluator:
    """CUDA-optimized TEDS evaluator with multi-GPU support"""
    
    def __init__(self, model, processor, template, device='cuda'):
        self.model = model
        self.processor = processor
        self.template = template
        self.device = device
        
        # Initialize TEDS with caching
        self.teds_scorer = TEDS(structure_only=False)
        self.teds_structure_scorer = TEDS(structure_only=True)
        
        # Cache for repeated evaluations
        self.prediction_cache = {}
        self.max_cache_size = 100
        
        print(f"✓ CUDA TEDS evaluator initialized on {device}")
    
    def generate_prediction(self, sample, max_new_tokens=None):
        """Generate HTML prediction with CUDA optimization"""
        if max_new_tokens is None:
            max_new_tokens = max_new_tokens_eval  # Use global eval token limit
            
        try:
            # Create cache key from sample
            cache_key = hash(str(sample.get('messages', [])))
            if cache_key in self.prediction_cache:
                return self.prediction_cache[cache_key]
            
            self.model.eval()
            
            with torch.no_grad():
                input_sample = {
                    'messages': [sample['messages'][0]],
                    'images': sample.get('images', [])
                }
                
                inputs = self.template.encode(input_sample)
                
                # Check input token count
                input_token_count = len(inputs['input_ids'])
                if input_token_count > max_length:
                    logger.warning(f"Input exceeds max_length: {input_token_count} > {max_length}")
                    # Truncate if needed
                    inputs['input_ids'] = inputs['input_ids'][:max_length]
                    if 'attention_mask' in inputs:
                        inputs['attention_mask'] = inputs['attention_mask'][:max_length]
                
                print(f"Input tokens: {input_token_count:,}, Max output: {max_new_tokens:,}")
                
                # Efficient tensor handling for CUDA
                input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long, device=self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                
                generation_kwargs = {
                    'use_cache': True,  # Enable cache for faster generation
                    'pad_token_id': self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
                }
                
                # Handle additional inputs efficiently
                for key, tensor_key in [('attention_mask', 'attention_mask'), 
                                      ('pixel_values', 'pixel_values'),
                                      ('image_grid_thw', 'image_grid_thw')]:
                    if key in inputs:
                        tensor = torch.tensor(inputs[key], device=self.device)
                        if tensor.dim() == 1 and key != 'pixel_values':
                            tensor = tensor.unsqueeze(0)
                        generation_kwargs[tensor_key] = tensor
                
                # Optimized generation parameters for CUDA
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    use_cache=True,
                    **generation_kwargs
                )
                
                input_length = input_ids.shape[1]
                generated_tokens = outputs[0][input_length:]
                prediction = self.processor.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                output_token_count = len(generated_tokens)
                print(f"Generated {output_token_count:,} output tokens")
                
                # Cache management
                if len(self.prediction_cache) >= self.max_cache_size:
                    oldest_key = next(iter(self.prediction_cache))
                    del self.prediction_cache[oldest_key]
                
                self.prediction_cache[cache_key] = prediction
                
                # Cleanup
                del input_ids, outputs, generated_tokens
                for tensor in generation_kwargs.values():
                    if isinstance(tensor, torch.Tensor):
                        del tensor
                
                return prediction
                
        except Exception as e:
            logger.warning(f"Generation error: {e}")
            return ""

# Cell 4: Multi-GPU optimized trainer
class MultiGPUTEDSTrainer(Seq2SeqTrainer):
    """Multi-GPU optimized trainer with enhanced memory management"""
    
    def __init__(self, teds_evaluator=None, **kwargs):
        super().__init__(**kwargs)
        self.teds_evaluator = teds_evaluator
        self.eval_count = 0
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """CUDA-optimized training step"""
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Efficient GPU memory management
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.args.device, non_blocking=True)
        
        result = super().training_step(model, inputs)
        
        # GPU memory cleanup
        if self.state.global_step % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        if self.state.global_step % 50 == 0:
            log_memory(f"Step {self.state.global_step}")
            
        return result
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Multi-GPU optimized evaluation"""
        log_memory("Before eval")
        
        print(f"\n{'='*40}")
        print(f"EVALUATION {self.eval_count + 1}")
        print(f"{'='*40}")
        
        # Standard evaluation for loss
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # TEDS evaluation
        if self.teds_evaluator:
            try:
                original_eval_dataset = getattr(self, 'eval_dataset_original', eval_dataset)
                if original_eval_dataset:
                    teds_results = self._evaluate_teds(original_eval_dataset)
                    eval_results.update(teds_results)
                else:
                    eval_results.update({
                        'eval_teds': 0.0,
                        'eval_teds_structure': 0.0,
                        'eval_teds_samples': 0
                    })
                    
            except Exception as e:
                logger.error(f"TEDS evaluation failed: {e}")
                eval_results.update({
                    'eval_teds': 0.0,
                    'eval_teds_structure': 0.0,
                    'eval_teds_samples': 0
                })
        
        self.eval_count += 1
        
        print(f"Results:")
        print(f"  Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
        print(f"  TEDS: {eval_results.get('eval_teds', 'N/A'):.4f}")
        print(f"  Structure: {eval_results.get('eval_teds_structure', 'N/A'):.4f}")
        print(f"  Samples: {eval_results.get('eval_teds_samples', 0)}")
        
        aggressive_cleanup()
        log_memory("After eval")
            
        return eval_results
    
    def _evaluate_teds(self, eval_dataset):
        """CUDA-optimized TEDS evaluation"""
        max_samples = min(5, len(eval_dataset))  # Increased samples for GPU
        print(f"TEDS evaluation on {max_samples} samples...")
        
        teds_scores = []
        structure_scores = []
        successful = 0
        
        for i in range(max_samples):
            try:
                sample = eval_dataset[i]
                
                # Generate prediction
                pred_html = self.teds_evaluator.generate_prediction(sample, max_new_tokens=max_new_tokens_eval)
                
                # Get ground truth
                if 'messages' in sample and len(sample['messages']) > 1:
                    true_html = sample['messages'][1]['content']
                elif 'html_table' in sample:
                    true_html = sample['html_table']
                else:
                    continue
                    
                # TEDS comparison
                teds_score = self.teds_evaluator.teds_scorer(true_html, pred_html)
                structure_score = self.teds_evaluator.teds_structure_scorer(true_html, pred_html)
                
                teds_scores.append(teds_score)
                structure_scores.append(structure_score)
                successful += 1
                
                print(f"  Sample {i+1}: TEDS={teds_score:.3f}, Structure={structure_score:.3f}")
                
            except Exception as e:
                print(f"  Error in sample {i+1}: {e}")
                continue
        
        avg_teds = np.mean(teds_scores) if teds_scores else 0.0
        avg_structure = np.mean(structure_scores) if structure_scores else 0.0
        
        return {
            'eval_teds': avg_teds,
            'eval_teds_structure': avg_structure,
            'eval_teds_samples': successful
        }

# Cell 5: Load dataset with validation split
logger.info("Loading dataset with validation split...")
dataset_name = "ruilin808/dataset_1920x1280"

# Load both training and validation splits
train_dataset = hf_load_dataset(dataset_name, split="train")
val_dataset = hf_load_dataset(dataset_name, split="validation")

logger.info(f"Training dataset: {len(train_dataset)} samples")
logger.info(f"Validation dataset: {len(val_dataset)} samples")

# Convert to ms-swift format
logger.info("Converting datasets to ms-swift format...")
train_processed = train_dataset.map(create_ms_swift_format_single, num_proc=4)
val_processed = val_dataset.map(create_ms_swift_format_single, num_proc=4)

logger.info("✓ Dataset conversion completed")

# Cell 6: Multi-GPU model loading
logger.info("Loading model with multi-GPU support...")
log_memory("Before model load")

# Calculate memory distribution across GPUs
if num_gpus > 1:
    # Create device map for multi-GPU
    device_map = {}
    total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus))
    
    # Distribute layers across GPUs
    device_map = "auto"  # Let transformers auto-distribute
    print(f"✓ Using automatic device mapping across {num_gpus} GPUs")
else:
    device_map = {"": 0}  # Single GPU
    print("✓ Using single GPU")

# Load model with multi-GPU support
model, processor = get_model_tokenizer(
    model_id_or_path=model_id_or_path,
    device_map=device_map,
    torch_dtype=torch.float16,  # Use FP16 for better GPU performance
    low_cpu_mem_usage=True,
    load_in_8bit=False,
    load_in_4bit=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
)

# Enable gradient checkpointing for memory efficiency
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

print(f"✓ Model loaded across GPUs")
log_memory("After model load")

# Template setup
template = get_template(model.model_meta.template, processor, default_system=system, max_length=max_length)
template.set_mode('train')
if template.use_model:
    template.model = model

print(f"✓ Template configured with max_length: {max_length:,} tokens")

# LoRA setup
linear_layers = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) and any(keyword in name.lower() for keyword in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
        linear_layers.append(name)

if linear_layers:
    target_modules = linear_layers
else:
    target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                freeze_aligner=freeze_aligner)

lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=target_modules)
model = Swift.prepare_model(model, lora_config)

logger.info("✓ LoRA setup completed")
model_parameter_info = get_model_parameter_info(model)
logger.info(f'Trainable parameters: {model_parameter_info}')

# Cell 7: Multi-GPU training arguments
output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'Output directory: {output_dir}')

# Calculate batch sizes based on available GPUs
per_device_batch_size = max(1, 8 // num_gpus)  # Adjust based on GPU count
gradient_accumulation_steps = max(1, 8 // (per_device_batch_size * num_gpus))

print(f"✓ Batch size per device: {per_device_batch_size}")
print(f"✓ Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"✓ Effective batch size: {per_device_batch_size * num_gpus * gradient_accumulation_steps}")

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    weight_decay=weight_decay,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=warmup_ratio,
    report_to=report_to,
    logging_first_step=logging_first_step,
    save_strategy='steps',
    save_steps=100,
    eval_strategy='steps',
    eval_steps=100,
    num_train_epochs=num_train_epochs,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    load_best_model_at_end=load_best_model_at_end,
    save_total_limit=3,
    logging_steps=10,
    dataloader_num_workers=4,  # Increased for better GPU utilization
    dataloader_pin_memory=True,  # Enable for faster GPU transfer
    data_seed=data_seed,
    remove_unused_columns=False,
    fp16=True,  # Enable FP16 for better GPU performance
    bf16=False,  # Use FP16 instead of BF16 for wider GPU support
    torch_compile=False,  # Can enable for newer PyTorch versions
    max_grad_norm=max_grad_norm,
    dataloader_drop_last=True,
    eval_accumulation_steps=1,
    save_safetensors=True,
    optim="adamw_torch_fused",  # Use fused optimizer for better GPU performance
    dataloader_persistent_workers=True,
    prediction_loss_only=False,
    include_inputs_for_metrics=False,
    ddp_find_unused_parameters=False,  # Optimize DDP
    ddp_backend="nccl" if num_gpus > 1 else None,  # Use NCCL for multi-GPU
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),  # Support for distributed training
)

# Cell 8: Dataset preparation
train_lazy_dataset = LazyLLMDataset(train_processed, template.encode, random_state=data_seed)
val_lazy_dataset = LazyLLMDataset(val_processed, template.encode, random_state=data_seed)

# Cell 9: Setup evaluator and trainer
# Get the main GPU device for evaluation
main_device = 'cuda:0' if torch.cuda.is_available() else 'cuda'

# Set up CUDA-optimized TEDS evaluator
teds_evaluator = CUDATEDSEvaluator(model, processor, template, device=main_device)

# Set up multi-GPU trainer
trainer = MultiGPUTEDSTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_lazy_dataset,
    eval_dataset=val_lazy_dataset,
    template=template,
    teds_evaluator=teds_evaluator,
)

trainer.eval_dataset_original = val_processed

# Enable input gradients
model.enable_input_require_grads()

log_memory("Ready for multi-GPU training")

# Cell 10: Execute training with GPU monitoring
logger.info("=" * 60)
logger.info("STARTING MULTI-GPU TRAINING")
logger.info("=" * 60)

# GPU memory monitoring
def gpu_memory_monitor():
    """Monitor GPU memory usage during training"""
    while True:
        try:
            log_memory("GPU Monitor")
            time.sleep(300)  # Check every 5 minutes
        except:
            break

# Start GPU monitoring thread
if torch.cuda.is_available():
    monitor_thread = threading.Thread(target=gpu_memory_monitor, daemon=True)
    monitor_thread.start()

try:
    trainer.train()
    logger.info("✓ Multi-GPU training completed successfully!")
    
except Exception as e:
    logger.error(f'Training failed: {e}')
    log_memory("Error state")
    raise

# Cell 11: Final cleanup and summary
logger.info("=" * 60)
logger.info("MULTI-GPU TRAINING SUMMARY")
logger.info("=" * 60)
logger.info(f"✓ Used {num_gpus} GPU(s)")
logger.info(f"✓ FP16 mixed precision training")
logger.info(f"✓ Batch size per device: {per_device_batch_size}")
logger.info(f"✓ Gradient accumulation: {gradient_accumulation_steps}")
logger.info(f"✓ Effective batch size: {per_device_batch_size * num_gpus * gradient_accumulation_steps}")
logger.info("✓ Flash Attention 2 (if available)")
logger.info("✓ Gradient checkpointing enabled")
logger.info("✓ Optimized data loading (4 workers)")
logger.info("✓ Persistent workers enabled")
logger.info("✓ NCCL backend for multi-GPU communication")

# Final memory status
log_memory("Final")

# Clear cache
if hasattr(teds_evaluator, 'prediction_cache'):
    teds_evaluator.prediction_cache.clear()
aggressive_cleanup()

logger.info("🚀 Multi-GPU training completed!")