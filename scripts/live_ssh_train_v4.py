import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("Phase 1: Loading Qwen2.5-0.5B-Instruct...")
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_id = "checkpoints/manglish-qwen-refined"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ensure pad stays away from assistant response tokens
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Phase 2: Loading dataset (50 curated samples)...")
dataset = load_dataset("json", data_files={"train": "data/manglish_booking_data.jsonl"}, split="train")
dataset = dataset.select(range(min(50, len(dataset)))) 

training_args = SFTConfig(
    dataset_text_field="text",
    output_dir=adapter_id,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5, # Significantly lower LR for stability
    logging_steps=10,
    optim="adafactor", # RAM efficient
    num_train_epochs=10, 
    report_to="none",
    use_cpu=True
)

trainer = SFTTrainer(model=model, train_dataset=dataset, processing_class=tokenizer, args=training_args)

print("Phase 3: Stable CPU Training (10 Epochs)...")
trainer.train()

print(f"Phase 4: Final Inference Test...")
prompt = "<|im_start|>system\nYou are StrawCore AI. Respond in Manglish/Malayalam.<|im_end|>\n<|im_start|>user\nEnikku kochi yil oru hotel room book cheyyanam, 2 perundu.<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

outputs = trainer.model.generate(
    **inputs, 
    max_new_tokens=50, 
    temperature=0.7, 
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
)

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
result = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
print(f"\n\n--- STABLE AI RESULT ---\n\n{result}\n\n------------------------\n")
