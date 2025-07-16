#!/usr/bin/env python3
"""
Memory-Optimized Qwen2-VL Fine-tuning Script for RunPod with /workspace storage
"""

import os
import torch
import gc
import json
import re
import shutil
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any
from pathlib import Path

# FIRST: Setup RunPod workspace before any other imports
def setup_runpod_workspace():
    """Setup RunPod workspace directories and environment variables"""
    print("Setting up RunPod workspace...")
    
    # Create necessary directories in /workspace
    workspace_dirs = [
        "/workspace/cache",
        "/workspace/cache/huggingface",
        "/workspace/cache/modelscope", 
        "/workspace/cache/datasets",
        "/workspace/models",
        "/workspace/datasets",
        "/workspace/output"
    ]
    
    for directory in workspace_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Set environment variables BEFORE importing other libraries
    os.environ['HF_HOME'] = '/workspace/cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/cache/huggingface/transformers'
    os.environ['HF_DATASETS_CACHE'] = '/workspace/cache/datasets'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/workspace/cache/huggingface/hub'
    os.environ['MODELSCOPE_CACHE'] = '/workspace/cache/modelscope'
    os.environ['TORCH_HOME'] = '/workspace/cache/torch'
    
    # Create symbolic links from default cache locations
    default_hf_cache = os.path.expanduser('~/.cache/huggingface')
    default_modelscope_cache = os.path.expanduser('~/.cache/modelscope')
    
    # Remove existing and create symlinks
    for default_cache, workspace_cache in [
        (default_hf_cache, '/workspace/cache/huggingface'),
        (default_modelscope_cache, '/workspace/cache/modelscope')
    ]:
        if os.path.exists(default_cache) and not os.path.islink(default_cache):
            shutil.rmtree(default_cache, ignore_errors=True)
        
        os.makedirs(os.path.dirname(default_cache), exist_ok=True)
        
        if not os.path.exists(default_cache):
            os.symlink(workspace_cache, default_cache)
            print(f"Created symlink: {default_cache} -> {workspace_cache}")
    
    print("RunPod workspace setup complete!")

# Setup workspace FIRST
setup_runpod_workspace()

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

# Setup GPUs after workspace setup
gpu_count = setup_gpus()

# Now import other modules
from swift.llm import (
    get_model_tokenizer, get_template,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrainerCallback
from datasets import load_dataset as hf_load_dataset

# TEDS evaluation imports
try:
    from table_recognition_metric import TEDS
    TEDS_AVAILABLE = True
    print("TEDS evaluation enabled")
except ImportError:
    TEDS_AVAILABLE = False
    print("WARNING: table_recognition_metric not installed. TEDS evaluation disabled.")

def download_to_workspace():
    """Download model and dataset to workspace if not already present"""
    model_id = 'Qwen/Qwen2.5-VL-32B-Instruct'
    dataset_name = "ruilin808/dataset_1920x1280"
    
    # Check if model exists in workspace
    model_workspace_path = f'/workspace/cache/modelscope/hub/models/{model_id}'
    if not os.path.exists(model_workspace_path):
        print(f"Downloading {model_id} to workspace...")
        
        # Import here to ensure environment variables are set
        from modelscope import snapshot_download
        
        try:
            snapshot_download(
                model_id, 
                cache_dir='/workspace/cache/modelscope'
            )
            print(f"Model downloaded to workspace cache")
        except Exception as e:
            print(f"Error downloading model: {e}")
    else:
        print(f"Model already exists in workspace: {model_workspace_path}")
    
    # Check if dataset exists
    dataset_cache_path = f'/workspace/cache/datasets/{dataset_name.replace("/", "_")}'
    if not os.path.exists(dataset_cache_path):
        print(f"Downloading {dataset_name} to workspace...")
        
        try:
            # This will use the HF_DATASETS_CACHE we set to /workspace/cache/datasets
            dataset = hf_load_dataset(dataset_name)
            print(f"Dataset cached to workspace")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
    else:
        print(f"Dataset already exists in workspace cache")

def check_workspace_storage():
    """Check available storage in workspace"""
    try:
        stat = shutil.disk_usage('/workspace')
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3) 
        free_gb = stat.free / (1024**3)
        
        print(f"\nWorkspace Storage Status:")
        print(f"  Total: {total_gb:.1f} GB")
        print(f"  Used:  {used_gb:.1f} GB") 
        print(f"  Free:  {free_gb:.1f} GB")
        print(f"  Usage: {(used_gb/total_gb)*100:.1f}%")
        
        if free_gb < 50:
            print("⚠️  WARNING: Low disk space!")
        
        return free_gb > 50
    except Exception as e:
        print(f"Error checking storage: {e}")
        return True

# [Keep all your existing functions like normalize_html_for_teds, evaluate_single_teds, etc.]
# ... (paste your existing helper functions here)

# Fixed TEDSEvaluationCallback (using the fixed version from previous artifact)
class TEDSEvaluationCallback(TrainerCallback):
    """Custom callback to compute TEDS scores during training"""
    
    def __init__(self, template, eval_dataset, eval_frequency=100, gpu_count=1):
        super().__init__()
        self.template = template
        self.eval_dataset = eval_dataset
        self.eval_frequency = eval_frequency
        self.gpu_count = gpu_count
        self.step_count = 0
        self.eval_count = 0
        self.teds_history = []
        
        # Scale sample sizes based on GPU count
        if gpu_count >= 8:
            self.quick_sample_size = min(50, len(eval_dataset))
            self.thorough_sample_size = min(150, len(eval_dataset))
        elif gpu_count >= 4:
            self.quick_sample_size = min(30, len(eval_dataset))
            self.thorough_sample_size = min(100, len(eval_dataset))
        else:
            self.quick_sample_size = min(20, len(eval_dataset))
            self.thorough_sample_size = min(75, len(eval_dataset))
        
        print(f"TEDS Evaluation Configuration:")
        print(f"  - Quick evaluation: {self.quick_sample_size} samples")
        print(f"  - Thorough evaluation: {self.thorough_sample_size} samples (every 5th evaluation)")
    
    def on_init_end(self, args, state, control, **kwargs):
        """Called at the end of trainer initialization"""
        print("TEDS Evaluation Callback initialized successfully")
        return control
    
    # [Include the rest of the fixed callback methods from the previous artifact]

def main():
    """Main training function with RunPod workspace integration"""
    
    logger = get_logger()
    seed_everything(42)
    
    # Check workspace storage first
    if not check_workspace_storage():
        logger.warning("Proceeding with low disk space - monitor usage carefully!")
    
    # Download model and dataset to workspace
    download_to_workspace()
    
    # Clear initial GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Configuration - Use workspace paths
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'  # Will use cached version
    output_dir = '/workspace/output'  # Save to workspace
    data_seed = 42
    
    # [Keep all your existing configuration and scaling logic]
    # ... (paste your existing configuration code here)
    
    # Load model - will use workspace cache
    logger.info("Loading model from workspace cache...")
    model, processor = get_model_tokenizer(
        model_id_or_path,  # This will now use the cached version
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    
    # [Continue with rest of your existing main() function]
    # ... (paste the rest of your training code here)
    
    # At the end, show final workspace usage
    print("\n" + "=" * 50)
    print("Final workspace storage status:")
    check_workspace_storage()
    
    logger.info(f'Training complete. Model saved to: {output_dir}')

if __name__ == "__main__":
    main()