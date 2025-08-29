from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)

# Prepare input
input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids

# Generate output
output_ids = model.generate(
    input_ids,
    max_length=30,
    num_beams=5,
    early_stopping=True
)

# Decode to text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)