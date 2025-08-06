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
    device_map="balanced",
)

"""
We now add LoRA adapters for parameter efficient finetuning - this allows us to only efficiently train 1% of all parameters.

**[NEW]** We also support finetuning ONLY the vision part of the model, or ONLY the language part. Or you can select both! 
You can also select to finetune the attention or the MLP layers!
"""

# Configure LoRA adapters with increased regularization
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,    # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,      # False if not finetuning MLP layers
    r=4,                          # The larger, the higher the accuracy, but might overfit
    lora_alpha=8,                 # Recommended alpha == r at least
    lora_dropout=0.2,             # INCREASED from 0.1 for better regularization
    bias="none",
    random_state=3407,
    use_rslora=False,              # We support rank stabilized LoRA
    loftq_config=None,             # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

"""
### Data Preparation with HTML Filtering
We'll be using the new dataset with custom train/validation split for small dataset optimization.
"""

from datasets import load_dataset

# Load only the training dataset (no separate validation available)
full_dataset = load_dataset("ruilin808/tables_1920_x_1280", split="train")

print(f"Full dataset size: {len(full_dataset)}")

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
        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples...")
    
    print(f"Filtered {dataset_name} dataset size: {len(filtered_samples)}")
    print(f"Removed {invalid_count} invalid samples from {dataset_name}")
    print(f"Filtering efficiency: {len(filtered_samples)/len(dataset)*100:.2f}%")
    print()
    
    return filtered_samples

# Apply filtering to the full dataset
filtered_full_dataset = apply_html_filtering(full_dataset, "full")

"""
### Create Train/Validation Split (85/15)
Split the filtered dataset into training and validation sets for small dataset optimization.
"""

import random

def create_train_val_split(dataset, train_ratio=0.85, random_seed=3407):
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: List of samples
        train_ratio: Proportion for training (0.85 = 85%)
        random_seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    random.seed(random_seed)
    
    # Shuffle the dataset
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    
    # Calculate split index
    total_samples = len(shuffled_dataset)
    train_size = int(total_samples * train_ratio)
    
    # Split the data
    train_dataset = shuffled_dataset[:train_size]
    val_dataset = shuffled_dataset[train_size:]
    
    print(f"Dataset split created:")
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {len(train_dataset)} ({len(train_dataset)/total_samples*100:.1f}%)")
    print(f"Validation samples: {len(val_dataset)} ({len(val_dataset)/total_samples*100:.1f}%)")
    print()
    
    return train_dataset, val_dataset

# Create the train/validation split
filtered_train_dataset, filtered_val_dataset = create_train_val_split(
    filtered_full_dataset, 
    train_ratio=0.85,
    random_seed=3407
)

"""
### Prepare Inference Samples from Validation Dataset
Sort validation dataset by html_table length (decreasing) and take top 10 samples for testing.
"""

def prepare_inference_samples(val_dataset, num_samples=10):
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
    
    # Take top samples (but don't exceed available samples)
    actual_num_samples = min(num_samples, len(sorted_samples))
    top_samples = sorted_samples[:actual_num_samples]
    
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

# Prepare inference samples (reduced to 10 for small validation set)
inference_samples = prepare_inference_samples(filtered_val_dataset, num_samples=10)

# Define instruction for the task
instruction = """Convert this table image to HTML format with proper formatting.

Requirements:
- Table tag: <table border='1' style='border-collapse: collapse; width: 100%;'>
- Use <th> tags for header rows, <td> for data cells
- Proper word spacing in cells
- Handle merged cells with appropriate colspan/rowspan attributes
- Preserve original table structure and formatting
- Use <sup></sup> for superscript text
- Keep special characters as-is (don't convert <, ≤, ≥ to HTML entities)
- Maintain exact cell content from image, including any grammar errors
- Do NOT preserve footnotes or references below the table

Output clean, properly formatted HTML table code. Do not include any additional text or explanations."""

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

print("\n" + "="*60)
print("PRE-TRAINING INFERENCE COMPLETED!")
print("="*60)
print("OUTPUT LOCATIONS:")
print("Ground truth HTML files saved in: html/")
print("Sample images saved in: images/")
print("Pre-training outputs saved in: pre_training_outputs/")
print("="*60)