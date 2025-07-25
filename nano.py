from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
import io

model_path = "nanonets/Nanonets-OCR-s"

model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    torch_dtype="auto", 
    device_map="auto", 
    attn_implementation="flash_attention_2"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)


def ocr_image_with_nanonets_s(image, model, processor, max_new_tokens=4096):
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]


# Load the dataset
dataset = load_dataset("ruilin808/dataset_1920x1280", split="validation")

# Filter rows that have html_table column and sort by string length
rows_with_html = [row for row in dataset if row.get('html_table') is not None]
rows_with_html.sort(key=lambda x: len(str(x['html_table'])), reverse=True)

# Get the longest 20 rows
longest_20_rows = rows_with_html[:20]

print(f"Found {len(rows_with_html)} rows with html_table data")
print(f"Processing the longest 20 rows...")

# Process each of the longest 20 rows
results = []
for i, row in enumerate(longest_20_rows):
    print(f"\nProcessing row {i+1}/20...")
    print(f"HTML table length: {len(str(row['html_table']))} characters")
    
    # Get the image from the row (assuming there's an 'image' column)
    # Adjust the column name if it's different in your dataset
    image = row['image']  # This should be a PIL Image or similar
    
    try:
        # Run OCR inference
        result = ocr_image_with_nanonets_s(image, model, processor, max_new_tokens=15000)
        
        # Store results
        results.append({
            'row_index': i,
            'html_table_length': len(str(row['html_table'])),
            'ground_truth_html': row['html_table'],
            'ocr_result': result
        })
        
        print(f"OCR completed for row {i+1}")
        print(f"Ground truth HTML (first 200 chars): {str(row['html_table'])[:200]}...")
        print(f"OCR result (first 200 chars): {result[:200]}...")
        
    except Exception as e:
        print(f"Error processing row {i+1}: {str(e)}")
        results.append({
            'row_index': i,
            'html_table_length': len(str(row['html_table'])),
            'ground_truth_html': row['html_table'],
            'ocr_result': f"Error: {str(e)}"
        })

# Print summary
print(f"\n=== PROCESSING COMPLETE ===")
print(f"Successfully processed {len([r for r in results if not r['ocr_result'].startswith('Error')])} out of 20 rows")

# Save results to individual HTML files and JSON summary
import json
import os

# Create directory for HTML files
os.makedirs('html_results', exist_ok=True)

for i, result in enumerate(results):
    if not result['ocr_result'].startswith('Error'):
        # Save ground truth HTML
        with open(f'html_results/row_{i+1:02d}_ground_truth.html', 'w', encoding='utf-8') as f:
            f.write(str(result['ground_truth_html']))
        
        # Save OCR result as HTML (if it contains HTML-like content)
        with open(f'html_results/row_{i+1:02d}_ocr_result.html', 'w', encoding='utf-8') as f:
            f.write(result['ocr_result'])
        
        print(f"Saved HTML files for row {i+1}")

# Also save summary JSON (without full HTML content to avoid duplication)
summary_results = []
for result in results:
    summary_results.append({
        'row_index': result['row_index'],
        'html_table_length': result['html_table_length'],
        'status': 'success' if not result['ocr_result'].startswith('Error') else 'error',
        'error_message': result['ocr_result'] if result['ocr_result'].startswith('Error') else None
    })

with open('ocr_results_summary.json', 'w') as f:
    json.dump(summary_results, f, indent=2)

print("Individual HTML files saved to 'html_results/' directory")
print("Summary saved to 'ocr_results_summary.json'")