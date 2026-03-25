from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

tokenizer = AutoTokenizer.from_pretrained("./tinyllama-base", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("./tinyllama-base", torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu", local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("./tinyllama-base", torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map="cpu", local_files_only=True)




lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files="data.jsonl", split="train")

def format_prompt(example):
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}<|endoftext|>"
    }

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    tokenized["labels"] = [
        [-100 if token_id == tokenizer.pad_token_id else token_id for token_id in labels] 
        for labels in tokenized["input_ids"]
    ]
    return tokenized

tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    batch_size=1000,
    remove_columns=dataset.column_names
)


training_args = TrainingArguments(
    output_dir="tinyllama-finetuned-fast",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch=8
    num_train_epochs=3,  # Fast test
    save_strategy="steps",
    save_steps=10,
    logging_steps=1,  # Frequent updates!
    learning_rate=2e-4,
    warmup_steps=5,
    dataloader_pin_memory=False,  # CPU fix
    fp16=False,  # CPU
    bf16=False,

    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,  # Stable CPU
    gradient_checkpointing=True,  # Memory efficient
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("🚀 Starting fast CPU training (ETA ~20-40min)...")
trainer.train()
trainer.model.save_pretrained("tinyllama-finetuned-fast")
tokenizer.save_pretrained("tinyllama-finetuned-fast")
print("✅ Training complete! Test with step4_test.py")