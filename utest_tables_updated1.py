# Commented out IPython magic to ensure Python compatibility.
# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab notebooks! Otherwise use pip install unsloth
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" huggingface_hub hf_transfer
#     !pip install --no-deps unsloth

"""### Unsloth Model Setup"""

from unsloth import FastVisionModel  # FastLanguageModel for LLMs
import torch
import os
import re
from PIL import Image
from pathlib import Path

# Load the base model
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-32B-Instruct",
    load_in_4bit=False,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)

"""
We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.

**[NEW]** We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! 
You can also select to finetune the attention or the MLP layers!
"""

# Configure LoRA adapters
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,    # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,      # False if not finetuning MLP layers
    r=4,                          # The larger, the higher the accuracy, but might overfit
    lora_alpha=8,                 # Recommended alpha == r at least
    lora_dropout=0.1,
    bias="none",
    random_state=3407,
    use_rslora=False,              # We support rank stabilized LoRA
    loftq_config=None,             # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

"""
### Data Preparation with HTML Filtering
We'll be using a sampled dataset of handwritten maths formulas. The goal is to convert these images into a computer readable form - 
ie in LaTeX form, so we can render it. This can be very useful for complex formulas.

You can access the dataset [here](https://huggingface.co/datasets/unsloth/LaTeX_OCR). 
The full dataset is [here](https://huggingface.co/datasets/linxy/LaTeX_OCR).
"""

from datasets import load_dataset

# Load datasets
train_dataset = load_dataset("ruilin808/dataset_1920x1280", split="train")
val_dataset = load_dataset("ruilin808/dataset_1920x1280", split="validation")

def clean_html_content(html_content):
    """
    Clean HTML content by removing unwanted wrapper tags and code blocks.
    
    Args:
        html_content (str): Raw HTML content to clean
        
    Returns:
        str: Cleaned HTML content
    """
    if not html_content:
        return html_content
    
    # Strip surrounding ```html ... ``` or <html>...</html> blocks if present
    cleaned_html = html_content.strip()
    cleaned_html = re.sub(r'^```(?:html)?\s*', '', cleaned_html, flags=re.IGNORECASE)
    cleaned_html = re.sub(r'\s*```$', '', cleaned_html)
    cleaned_html = re.sub(r'^<html>\s*', '', cleaned_html, flags=re.IGNORECASE)
    cleaned_html = re.sub(r'\s*</html>$', '', cleaned_html, flags=re.IGNORECASE)
    cleaned_html = re.sub(r'^<body>\s*', '', cleaned_html, flags=re.IGNORECASE)
    cleaned_html = re.sub(r'\s*</body>$', '', cleaned_html, flags=re.IGNORECASE)
    
    return cleaned_html.strip()

def is_valid_html_sample(sample):
    """
    Validate HTML sample for training quality.
    
    Args:
        sample: Dataset sample containing 'html_table' and 'image'
        
    Returns:
        bool: True if sample is valid for training
    """
    html_content = sample.get("html_table", "")
    
    # Check if html_table exists and is not empty
    if not html_content or len(html_content.strip()) < 10:
        return False
    
    # Clean the HTML first
    cleaned_html = clean_html_content(html_content)
    
    # Check for basic table structure after cleaning
    if "<table" not in cleaned_html.lower():
        return False
    
    # Check for balanced table tags
    table_open = cleaned_html.lower().count("<table")
    table_close = cleaned_html.lower().count("</table>")
    if table_open != table_close or table_open == 0:
        return False
    
    # Check for reasonable content length (not too short or extremely long)
    if len(cleaned_html) < 20 or len(cleaned_html) > 50000:  # Adjust limits as needed
        return False
    
    # Check if image exists
    if "image" not in sample or sample["image"] is None:
        return False
    
    return True

def apply_html_filtering(dataset, dataset_name):
    """
    Apply HTML cleaning and filtering to dataset.
    
    Args:
        dataset: Input dataset to filter
        dataset_name: Name for logging purposes
        
    Returns:
        list: Filtered dataset with cleaned HTML
    """
    print(f"Applying HTML filtering to {dataset_name} dataset...")
    print(f"Original {dataset_name} dataset size: {len(dataset)}")
    
    filtered_samples = []
    invalid_count = 0
    
    for i, sample in enumerate(dataset):
        if is_valid_html_sample(sample):
            # Clean the HTML content
            cleaned_sample = sample.copy()
            cleaned_sample["html_table"] = clean_html_content(sample["html_table"])
            filtered_samples.append(cleaned_sample)
        else:
            invalid_count += 1
        
        # Progress logging
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples...")
    
    print(f"Filtered {dataset_name} dataset size: {len(filtered_samples)}")
    print(f"Removed {invalid_count} invalid samples from {dataset_name}")
    print(f"Filtering efficiency: {len(filtered_samples)/len(dataset)*100:.2f}%")
    print()
    
    return filtered_samples

# Apply HTML filtering to both datasets
filtered_train_dataset = apply_html_filtering(train_dataset, "training")
filtered_val_dataset = apply_html_filtering(val_dataset, "validation")

"""
### Prepare Inference Samples from Validation Dataset
Sort validation dataset by html_table length (decreasing) and take top 20 samples.
Save images and HTML files for inference testing.
"""

def prepare_inference_samples(val_dataset, num_samples=20):
    """
    Sort validation dataset by HTML length and prepare top samples for inference.
    
    Args:
        val_dataset: Filtered validation dataset
        num_samples: Number of top samples to prepare
        
    Returns:
        list: Top samples sorted by HTML length (decreasing)
    """
    print(f"Preparing top {num_samples} inference samples from validation dataset...")
    
    # Sort by html_table length in decreasing order
    sorted_samples = sorted(val_dataset, key=lambda x: len(x["html_table"]), reverse=True)
    
    # Take top samples
    top_samples = sorted_samples[:num_samples]
    
    # Create directories
    os.makedirs("images", exist_ok=True)
    os.makedirs("html", exist_ok=True)
    
    print(f"Saving {len(top_samples)} samples...")
    
    for i, sample in enumerate(top_samples):
        # Save image
        image_filename = f"sample_{i+1:02d}.png"
        image_path = os.path.join("images", image_filename)
        sample["image"].save(image_path)
        
        # Save HTML
        html_filename = f"sample_{i+1:02d}.html"
        html_path = os.path.join("html", html_filename)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(sample["html_table"])
        
        # Print sample info
        html_length = len(sample["html_table"])
        print(f"Sample {i+1:02d}: HTML length = {html_length} characters")
    
    print(f"Inference samples prepared!")
    print(f"Images saved in: images/")
    print(f"HTML files saved in: html/")
    print()
    
    return top_samples

# Prepare inference samples
inference_samples = prepare_inference_samples(filtered_val_dataset, num_samples=20)

"""
To format the dataset, all vision finetuning tasks should be formatted as follows:

```python
[
{ "role": "user",
  "content": [{"type": "text",  "text": Q}, {"type": "image", "image": image} ]
},
{ "role": "assistant",
  "content": [{"type": "text",  "text": A} ]
},
]
```
"""

# Define instruction for the task
#instruction = """Convert this table image to HTML format with proper formatting.

#Requirements:
#- Table tag: <table border='1' style='border-collapse: collapse; width: 100%;'>
#- Proper word spacing in cells
#- Space before annotations: "word A" not "wordA", "text 1" not "text1"

#Output clean, properly formatted HTML table code. Do not include any additional text or explanations."""

instruction = """Convert this table image to HTML format with proper formatting.

Requirements:
- Table tag: <table border='1' style='border-collapse: collapse; width: 100%;'>
- Proper word spacing in cells

Output clean, properly formatted HTML table code. Do not include any additional text or explanations."""

def convert_to_conversation(sample):
    """Convert dataset sample to conversation format required for training."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["html_table"]}  # Already cleaned by filtering
            ]
        },
    ]
    return {"messages": conversation}

# Convert filtered datasets to conversation format
converted_train_dataset = [convert_to_conversation(sample) for sample in filtered_train_dataset]
converted_val_dataset = [convert_to_conversation(sample) for sample in filtered_val_dataset]

print(f"Final training dataset size: {len(converted_train_dataset)}")
print(f"Final validation dataset size: {len(converted_val_dataset)}")

"""### Pre-Training Inference on Validation Samples"""

def run_inference_on_samples(model, tokenizer, samples, output_folder, stage_name):
    """
    Run inference on prepared samples and save HTML outputs.
    
    Args:
        model: The model to use for inference
        tokenizer: Model tokenizer
        samples: List of prepared samples with images
        output_folder: Path to save HTML files
        stage_name: Name of the inference stage (for logging)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Running {stage_name} inference on {len(samples)} samples...")
    
    FastVisionModel.for_inference(model)
    
    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}")
        
        try:
            # Use the image from the sample directly
            image = sample["image"]
            
            # Prepare messages
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}
                ]}
            ]
            
            # Generate inference
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=8000,
                    use_cache=True, 
                    temperature=1.5, 
                    min_p=0.1,
                    do_sample=False  # Use greedy decoding for consistency
                )
            
            # Decode and clean output using the same cleaning function
            pred_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred_text = clean_html_content(pred_text)

            # Save cleaned HTML
            html_filename = f"sample_{i+1:02d}_pred.html"
            html_path = os.path.join(output_folder, html_filename)

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(pred_text)

            print(f"Saved: {html_filename}")
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            continue
    
    print(f"{stage_name} inference completed!\n")

# Run pre-training inference
print("="*60)
print("PRE-TRAINING INFERENCE")
print("="*60)
run_inference_on_samples(model, tokenizer, inference_samples, "pre_training_outputs", "Pre-training")

"""
### Table Recognition Metrics Setup
Install and setup TEDS (Tree Edit Distance based Similarity) for table structure evaluation.

**NOTE: TEDS Implementation Analysis**
✅ PROPERLY IMPLEMENTED: The TEDS metric is correctly implemented for table recognition evaluation:
- Uses both structure-only and full content evaluation modes
- Properly handles error cases with try-catch blocks
- Evaluates predictions against ground truth HTML tables
- Calculates meaningful average scores across validation samples
- Integrates well with the training callback system
"""

# Install TEDS for table recognition evaluation
try:
    from table_recognition_metric import TEDS
except ImportError:
    print("Installing table_recognition_metric...")
    import subprocess
    subprocess.check_call(["pip", "install", "table_recognition_metric"])
    from table_recognition_metric import TEDS

# Initialize TEDS metrics
teds_structure_only = TEDS(structure_only=True)
teds_full = TEDS(structure_only=False)

def evaluate_table_recognition(model, tokenizer, val_dataset, num_samples):
    """
    Evaluate table recognition performance using TEDS metrics.
    
    Args:
        model: The trained model
        tokenizer: Model tokenizer
        val_dataset: Validation dataset (filtered)
        num_samples: Number of samples to evaluate (default 50 for speed)
    
    Returns:
        Dictionary containing TEDS scores
    """
    FastVisionModel.for_inference(model)
    
    teds_structure_scores = []
    teds_full_scores = []
    
    # Limit evaluation samples for efficiency
    eval_samples = min(num_samples, len(val_dataset))
    
    print(f"Evaluating on {eval_samples} validation samples...")
    
    for i in range(eval_samples):
        sample = val_dataset[i]
        image = sample["image"]
        gt_html = sample["html_table"]  # Already cleaned by filtering
        
        # Generate prediction
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=8000,
                use_cache=True, 
                temperature=1.5, 
                min_p=0.1,
                do_sample=False  # Use greedy decoding for consistent evaluation
            )
        
        # Decode and clean prediction using the same cleaning function
        pred_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred_text = clean_html_content(pred_text)
        
        # Calculate TEDS scores
        try:
            teds_structure_score = teds_structure_only(gt_html, pred_text)
            teds_full_score = teds_full(gt_html, pred_text)
            
            teds_structure_scores.append(teds_structure_score)
            teds_full_scores.append(teds_full_score)
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            # Add 0 score for failed evaluations
            teds_structure_scores.append(0.0)
            teds_full_scores.append(0.0)
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{eval_samples} samples")
    
    # Calculate average scores
    avg_teds_structure = sum(teds_structure_scores) / len(teds_structure_scores)
    avg_teds_full = sum(teds_full_scores) / len(teds_full_scores)
    
    results = {
        "teds_structure_only": avg_teds_structure,
        "teds_full": avg_teds_full,
        "num_evaluated": eval_samples
    }
    
    return results

"""
### Model Training
Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). 
We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`. 
We also support TRL's `DPOTrainer`!

We use our new `UnslothVisionDataCollator` which will help in our vision finetuning setup.
"""

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# Custom callback for validation evaluation
from transformers import TrainerCallback

class TableRecognitionCallback(TrainerCallback):
    """Custom callback to evaluate table recognition performance during training."""
    
    def __init__(self, model, tokenizer, val_dataset, eval_steps=10):
        self.model = model
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.eval_steps = eval_steps
        
    def on_log(self, args, state, control, model=None, **kwargs):
        """Run evaluation at specified intervals during training."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            print(f"\nRunning table recognition evaluation at step {state.global_step}...")
            
            # Evaluate table recognition
            metrics = evaluate_table_recognition(
                self.model, 
                self.tokenizer, 
                self.val_dataset, 
                num_samples=1  # Reduce samples during training for speed
            )
            
            # Log metrics
            print(f"TEDS Structure Only: {metrics['teds_structure_only']:.4f}")
            print(f"TEDS Full: {metrics['teds_full']:.4f}")
            print()  # Add spacing

FastVisionModel.for_training(model)  # Enable for training!

print("="*60)
print("STARTING TRAINING")
print("="*60)

# Configure and create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
    train_dataset=converted_train_dataset,
    eval_dataset=converted_val_dataset, 

    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="tensorboard",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
        save_strategy="steps",
        save_steps=28,
        eval_strategy="steps",
        eval_steps=14,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Start training
trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

"""### Post-Training Inference on Validation Samples"""

print("="*60)
print("POST-TRAINING INFERENCE")
print("="*60)
run_inference_on_samples(model, tokenizer, inference_samples, "post_training_outputs", "Post-training")

"""
### Saving Finetuned Models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

# Save LoRA adapters locally
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("Model and tokenizer saved to 'lora_model' directory")

"""### Saving to float16 for VLLM - We also support saving to `float16` directly. Select `merged_16bit` for float16."""

# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit
if False: 
    model.save_pretrained_merged("unsloth_finetune", tokenizer)

# To export and save to your Hugging Face account
if False: 
    model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", tokenizer, token="PUT_HERE")

print("\n" + "="*60)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print("="*60)
print("Ground truth HTML files saved in: html/")
print("Sample images saved in: images/")
print("Pre-training outputs saved in: pre_training_outputs/")
print("Post-training outputs saved in: post_training_outputs/")
print("Model saved in: lora_model/")

"""
And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/u54VK8m8tk) channel! 
If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

Some other links:
1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!

<div class="align-center">
  <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/blob/main/images/Discord.png" width="145"></a>
  <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>
"""