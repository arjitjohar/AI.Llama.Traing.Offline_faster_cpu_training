from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_id) #download model weights from the internet
tokenizer = AutoTokenizer.from_pretrained(model_id) #tokenizes it.

model.save_pretrained("tinyllama-base") #saves base model to root directory 
tokenizer.save_pretrained("tinyllama-base")
print("✅ Base model downloaded to ./tinyllama-base")
