#!/usr/bin/env python3
"""
Minimal diagnostic test to identify the exact cause of token validation errors
"""

import torch
import os
from swift.llm import get_model_tokenizer, get_template
from datasets import load_dataset as hf_load_dataset

# Enable CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def test_single_sample_encoding():
    """Test encoding of a single sample to isolate the issue"""
    
    print("=== LOADING MODEL ===")
    model, processor = get_model_tokenizer(
        'Qwen/Qwen2.5-VL-32B-Instruct',
        torch_dtype=torch.float16,
        device_map='auto',
    )
    
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    template = get_template(model.model_meta.template, processor, default_system=None, max_length=8192)
    template.set_mode('train')
    template.model = model
    
    print("=== LOADING DATA ===")
    raw_dataset = hf_load_dataset("ruilin808/dataset_1920x1280")
    first_sample = raw_dataset['train'][0]
    
    print(f"HTML length: {len(first_sample['html_table'])}")
    print(f"HTML preview: {first_sample['html_table'][:200]}...")
    
    # Create Swift format
    swift_sample = {
        'messages': [
            {
                'role': 'user',
                'content': 'Write the HTML representation for this image of a medical table.'
            },
            {
                'role': 'assistant',
                'content': first_sample['html_table']
            }
        ],
        'images': [first_sample['image']]
    }
    
    print("=== TESTING TOKENIZER DIRECTLY ===")
    try:
        # Test tokenizer directly on HTML content
        html_tokens = tokenizer.encode(first_sample['html_table'])
        print(f"Direct tokenization successful: {len(html_tokens)} tokens")
        print(f"Token range: [{min(html_tokens)}, {max(html_tokens)}]")
        print(f"Valid range: [0, {tokenizer.vocab_size-1}]")
        
        if max(html_tokens) >= tokenizer.vocab_size:
            print("❌ FOUND THE ISSUE: Direct tokenization produces invalid tokens!")
            invalid_tokens = [t for t in html_tokens if t >= tokenizer.vocab_size]
            print(f"Invalid tokens: {invalid_tokens[:10]}...")  # Show first 10
            return
            
    except Exception as e:
        print(f"❌ Direct tokenization failed: {e}")
        return
    
    print("=== TESTING SWIFT TEMPLATE ENCODING ===")
    try:
        encoded = template.encode(swift_sample)
        input_ids = encoded['input_ids']
        labels = encoded.get('labels', None)
        
        print(f"Template encoding successful")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Input IDs range: [{input_ids.min()}, {input_ids.max()}]")
        
        if input_ids.max() >= tokenizer.vocab_size:
            print("❌ FOUND THE ISSUE: Template encoding produces invalid input_ids!")
            return
            
        if labels is not None:
            print(f"Labels shape: {labels.shape}")
            valid_mask = labels != -100
            if valid_mask.any():
                valid_labels = labels[valid_mask]
                print(f"Valid labels range: [{valid_labels.min()}, {valid_labels.max()}]")
                
                if valid_labels.max() >= tokenizer.vocab_size:
                    print("❌ FOUND THE ISSUE: Template encoding produces invalid labels!")
                    return
            else:
                print("All labels are -100 (ignored)")
        
    except Exception as e:
        print(f"❌ Template encoding failed: {e}")
        return
    
    print("=== TESTING MODEL FORWARD PASS ===")
    try:
        # Test single forward pass
        model.eval()
        with torch.no_grad():
            # Prepare inputs
            inputs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            print("Input shapes:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}, dtype: {v.dtype}, device: {v.device}")
                    if 'ids' in k.lower() or 'labels' in k.lower():
                        print(f"    range: [{v.min()}, {v.max()}]")
            
            outputs = model(**inputs)
            print(f"✅ Forward pass successful! Loss: {outputs.loss}")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    print("=== TESTING LOSS COMPUTATION SPECIFICALLY ===")
    try:
        # Test loss computation with different configurations
        if labels is not None:
            # Manually test loss computation
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            print(f"Logits shape: {logits.shape}")
            print(f"Shift logits shape: {shift_logits.shape}")
            print(f"Shift labels shape: {shift_labels.shape}")
            print(f"Shift labels range: [{shift_labels.min()}, {shift_labels.max()}]")
            
            # Check for problematic labels
            valid_mask = shift_labels != -100
            if valid_mask.any():
                valid_shift_labels = shift_labels[valid_mask]
                print(f"Valid shift labels range: [{valid_shift_labels.min()}, {valid_shift_labels.max()}]")
                
                if valid_shift_labels.max() >= tokenizer.vocab_size or valid_shift_labels.min() < 0:
                    print("❌ FOUND THE ISSUE: Labels after shifting are invalid!")
                    bad_labels = valid_shift_labels[(valid_shift_labels >= tokenizer.vocab_size) | (valid_shift_labels < 0)]
                    print(f"Bad label values: {bad_labels[:10].tolist()}")
                    return
            
            # Try manual loss computation
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            print(f"✅ Manual loss computation successful: {loss}")
            
    except Exception as e:
        print(f"❌ Manual loss computation failed: {e}")
        return
    
    print("✅ ALL TESTS PASSED - The issue might be elsewhere!")

if __name__ == "__main__":
    test_single_sample_encoding()