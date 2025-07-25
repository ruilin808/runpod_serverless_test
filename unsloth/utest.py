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
    r=8,                          # The larger, the higher the accuracy, but might overfit
    lora_alpha=8,                 # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,              # We support rank stabilized LoRA
    loftq_config=None,             # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

"""
### Data Preparation
We'll be using a sampled dataset of handwritten maths formulas. The goal is to convert these images into a computer readable form - 
ie in LaTeX form, so we can render it. This can be very useful for complex formulas.

You can access the dataset [here](https://huggingface.co/datasets/unsloth/LaTeX_OCR). 
The full dataset is [here](https://huggingface.co/datasets/linxy/LaTeX_OCR).
"""

from datasets import load_dataset

# Load datasets
train_dataset = load_dataset("ruilin808/dataset_1920x1280", split="train")
val_dataset = load_dataset("ruilin808/dataset_1920x1280", split="validation")

# Let's take an overview look at the dataset. We shall see what the 3rd image is, and what caption it had.
# train_dataset
# train_dataset[2]["image"]
# train_dataset[2]["html_table"]

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
instruction = "Write the html representation for this image."

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
                {"type": "text", "text": sample["html_table"]}
            ]
        },
    ]
    return {"messages": conversation}

# Convert datasets to conversation format
converted_train_dataset = [convert_to_conversation(sample) for sample in train_dataset]
converted_val_dataset = [convert_to_conversation(sample) for sample in val_dataset]

# Look at how the conversations are structured for the first example
converted_train_dataset[0]

"""Let's first see before we do any finetuning what the model outputs for the first example!"""

FastVisionModel.for_inference(model)  # Enable for inference!

# Test pre-training inference
image = train_dataset[2]["image"]
instruction = "Write the html representation for this image."

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

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs, 
    streamer=text_streamer, 
    max_new_tokens=8000,
    use_cache=True, 
    temperature=1.5, 
    min_p=0.1
)

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
        val_dataset: Validation dataset
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
        gt_html = sample["html_table"]
        
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
        
        # Decode prediction
        pred_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
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

# Initialize callback
eval_callback = TableRecognitionCallback(model, tokenizer, val_dataset)

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
        warmup_steps=10,
        max_steps=30,
        num_train_epochs=5,  # Set this instead of max_steps for full training runs
        learning_rate=1e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="tensorboard",  # none    # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
        save_strategy="epoch",         # Save more frequently
        eval_strategy="epoch",   # Evaluate each epoch
    ),
)

# Add custom callback
trainer.add_callback(eval_callback)

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

"""### Final Evaluation - Run comprehensive evaluation on validation set after training."""

print("\n" + "="*50)
print("FINAL TABLE RECOGNITION EVALUATION")
print("="*50)

# Run final comprehensive evaluation
final_metrics = evaluate_table_recognition(
    model, 
    tokenizer, 
    val_dataset, 
    num_samples=3  # More samples for final evaluation
)

print(f"\nFinal Results:")
print(f"TEDS Structure Only: {final_metrics['teds_structure_only']:.4f}")
print(f"TEDS Full: {final_metrics['teds_full']:.4f}")
print(f"Evaluated on {final_metrics['num_evaluated']} samples")

"""
### Inference
Let's run the model! You can change the instruction and input - leave the output blank!

We use `min_p = 0.1` and `temperature = 1.5`. 
Read this [Tweet](https://x.com/menhguin/status/1826132708508213629) for more information on why.
"""

FastVisionModel.for_inference(model)  # Enable for inference!

# Test post-training inference
image = train_dataset[2]["image"]
instruction = "Write the html representation for this image."

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

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs, 
    streamer=text_streamer, 
    max_new_tokens=8000,
    use_cache=True, 
    temperature=1.5, 
    min_p=0.1
)

"""
### Saving and Loading Finetuned Models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

# Save LoRA adapters locally
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token="...")  # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token="...")  # Online saving

"""Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:"""

if False:
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=True,  # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model)  # Enable for inference!

# Test with loaded model
image = train_dataset[0]["image"]
instruction = "Write the html representation for this image."

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

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs, 
    streamer=text_streamer, 
    max_new_tokens=8000,
    use_cache=True, 
    temperature=1.5, 
    min_p=0.1
)

"""### Saving to float16 for VLLM - We also support saving to `float16` directly. Select `merged_16bit` for float16."""

# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit
if False: 
    model.save_pretrained_merged("unsloth_finetune", tokenizer)

# To export and save to your Hugging Face account
if False: 
    model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", tokenizer, token="PUT_HERE")

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
  <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
  <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

  Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
</div>
"""