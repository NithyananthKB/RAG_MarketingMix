from sentence_transformers import SentenceTransformer, util
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read and extract text from PDF
pdf_path = '/Users/nithyananthkb/Downloads/tatamotors.pdf'
full_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        full_text += page.extract_text() or ""

# Chunk the text (e.g., 1000 characters per chunk)
chunk_size = 1000
chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

# Encode the chunks
doc_embeddings = model.encode(chunks, convert_to_tensor=True)

# Query
query = "What is the Consolidated revenue of the Tatamotors from operations?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute cosine similarities
cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

# Find the most similar chunk
top_result = cos_scores.argmax().item()
print(f"Most relevant chunk:\n{chunks[top_result]}")

# Load Llama model and tokenizer (replace with your model path or Hugging Face model name)
llama_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llm = AutoModelForCausalLM.from_pretrained(llama_model_name, torch_dtype=torch.float16, device_map="auto")

# Prepare prompt with context and question
context = chunks[top_result]
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
output = llm.generate(**inputs, max_new_tokens=200)
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n--- Llama LLM Answer ---")
print(answer)
