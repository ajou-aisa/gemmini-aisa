import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Prepare sequences
sequences = ["What is the population of the United States?", "What is the capital city of Japan?"]

## HS add
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token


# Tokenize input
encoded_input = tokenizer(sequences, return_tensors='pt', padding=True)
input_ids = encoded_input['input_ids']
print(input_ids)



# Generate text
generated_ids = model.generate(input_ids=input_ids, max_length=25)
generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]

# Print generated text
for i, text in enumerate(generated_texts):
    print(f"Generated text {i+1}: {text}\n")