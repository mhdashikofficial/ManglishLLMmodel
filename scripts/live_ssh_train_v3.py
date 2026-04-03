import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("Phase 1: Loading Qwen2.5-1.5B (Non-Gated)...")
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "checkpoints/manglish-qwen-1.5b"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Important for stop tokens
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Phase 2: Loading dataset...")
dataset = load_dataset("json", data_files={"train": "data/manglish_booking_data.jsonl"}, split="train")
dataset = dataset.select(range(min(100, len(dataset)))) 

training_args = SFTConfig(
    dataset_text_field="text",
    output_dir=adapter_id,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=0.001, 
    logging_steps=10,
    optim="adamw_torch",
    num_train_epochs=5,
    report_to="none",
    use_cpu=True
)

trainer = SFTTrainer(model=model, train_dataset=dataset, processing_class=tokenizer, args=training_args)

print("Phase 3: Deep Training (EPYC CPU)...")
trainer.train()

print(f"Phase 4: Saving {adapter_id}...")
trainer.model.save_pretrained(adapter_id)

print("\n--- INFERENCE TEST ---")
prompt = "<|im_start|>system\nYou are StrawCore AI. Respond to the client.<|im_end|>\n<|im_start|>user\nEnikku kochi yil oru hotel room book cheyyanam, 2 perundu.<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
outputs = trainer.model.generate(**inputs, max_new_tokens=40, temperature=0.3, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0], skip_special_tokens=True).split('assistant')[-1].strip()
print(f"OUTPUT: {response}")
