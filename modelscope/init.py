#!/usr/bin/env python3
"""
Qwen2-VL Fine-tuning Script
Fine-tunes Qwen2-VL-2B-Instruct model on LaTeX OCR dataset using LoRA.
"""

import os
from functools import partial
from PIL import Image

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, load_dataset, get_template, EncodePreprocessor, get_model_arch,
    get_multimodal_target_regex, LazyLLMDataset
)
from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def main():
    """Main training function"""
    
    # Initialize logger and set random seed
    logger = get_logger()
    seed_everything(42)
    
    # Hyperparameters for training
    # Model configuration
    model_id_or_path = 'Qwen/Qwen2-VL-2B-Instruct'
    system = None  # Using the default system defined in the template.
    output_dir = 'output'
    
    # Dataset configuration
    dataset = ['AI-ModelScope/LaTeX_OCR#20000']  # dataset_id or dataset_path. Sampling 20000 data points
    data_seed = 42
    max_length = 2048
    split_dataset_ratio = 0.01  # Split validation set
    num_proc = 4  # The number of processes for data loading.
    
    # LoRA configuration
    lora_rank = 8
    lora_alpha = 32
    freeze_llm = False
    freeze_vit = True
    freeze_aligner = True
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        report_to=['tensorboard'],
        logging_first_step=True,
        save_strategy='steps',
        save_steps=50,
        eval_strategy='steps',
        eval_steps=50,
        gradient_accumulation_steps=16,
        # To observe the training results more quickly, this is set to 1 here. 
        # Under normal circumstances, a larger number should be used.
        num_train_epochs=1,
        metric_for_best_model='loss',
        save_total_limit=5,
        logging_steps=5,
        dataloader_num_workers=4,
        data_seed=data_seed,
        remove_unused_columns=False,
    )
    
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    logger.info(f'output_dir: {output_dir}')
    
    # Create visual_loss directory
    visual_loss_dir = os.path.join(output_dir, 'visual_loss')
    os.makedirs(visual_loss_dir, exist_ok=True)
    logger.info(f'visual_loss_dir: {visual_loss_dir}')
    
    # Obtain the model and template
    model, processor = get_model_tokenizer(model_id_or_path)
    logger.info(f'model_info: {model.model_info}')
    template = get_template(model.model_meta.template, processor, default_system=system, max_length=max_length)
    template.set_mode('train')
    if template.use_model:
        template.model = model
    
    # Get target_modules and add trainable LoRA modules to the model.
    target_modules = get_multimodal_target_regex(model, freeze_llm=freeze_llm, freeze_vit=freeze_vit, 
                                freeze_aligner=freeze_aligner)
    lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
                             target_modules=target_modules)
    model = Swift.prepare_model(model, lora_config)
    logger.info(f'lora_config: {lora_config}')
    
    # Print model structure and trainable parameters.
    logger.info(f'model: {model}')
    model_parameter_info = get_model_parameter_info(model)
    logger.info(f'model_parameter_info: {model_parameter_info}')
    
    # Download and load the dataset, split it into a training set and a validation set,
    # and encode the text data into tokens.
    train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
                                              seed=data_seed)
    
    logger.info(f'train_dataset: {train_dataset}')
    logger.info(f'val_dataset: {val_dataset}')
    logger.info(f'train_dataset[0]: {train_dataset[0]}')
    
    train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=data_seed)
    val_dataset = LazyLLMDataset(val_dataset, template.encode, random_state=data_seed)
    data = train_dataset[0]
    logger.info(f'encoded_train_dataset[0]: {data}')
    
    template.print_inputs(data)
    
    # Get the trainer and start the training.
    model.enable_input_require_grads()  # Compatible with gradient checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=template.data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        template=template,
    )
    trainer.train()
    
    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    
    # Visualize the training loss and save to visual_loss folder
    # You can also use the TensorBoard visualization interface during training by entering
    # `tensorboard --logdir '{output_dir}/runs'` at the command line.
    logger.info(f'Saving loss visualization to: {visual_loss_dir}')
    plot_images(visual_loss_dir, training_args.logging_dir, ['train/loss'], 0.9)  # save images
    
    # Log the location of the saved loss image
    loss_image_path = os.path.join(visual_loss_dir, 'train_loss.png')
    if os.path.exists(loss_image_path):
        logger.info(f'Training loss visualization saved to: {loss_image_path}')
        logger.info('The light yellow line represents the actual loss value, '
                   'while the yellow line represents the loss value smoothed with a smoothing factor of 0.9.')
    else:
        logger.warning(f'Loss visualization image not found at: {loss_image_path}')


if __name__ == "__main__":
    main()