import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("Phase 1: Loading Base CPU Tensors...")
base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_model_id = "checkpoints/manglish-qwen"

model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cpu", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Phase 2: Loading dataset and filtering constraints...")
dataset = load_dataset("json", data_files={"train": "data/manglish_booking_data.jsonl"}, split="train")
dataset = dataset.select(range(min(40, len(dataset))))  # Use 40 samples to forcefully prove concept

training_args = SFTConfig(
    dataset_text_field="text",
    output_dir=adapter_model_id,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=0.002, 
    logging_steps=5,
    optim="adamw_torch",
    num_train_epochs=5,
    report_to="none",
    use_cpu=True
)

trainer = SFTTrainer(
    model=model, train_dataset=dataset, processing_class=tokenizer, args=training_args
)

print("Phase 3: Deep CPU PyTorch Execution!")
trainer.train()

print(f"Phase 4: Writing PEFT weights to {adapter_model_id}...")
trainer.model.save_pretrained(adapter_model_id)

print("\n-----------------------------------")
print("Executing GENERATION on CPU (please wait)...")

prompt = "<|im_start|>system\nYou are StrawCore AI, a highly capable assistant specializing in handling client bookings, consultations, and payments. You understand English, phonetic Manglish, and Malayalam natively.<|im_end|>\n<|im_start|>user\nEnikku kochi yil oru hotel room book cheyyanam, 2 perundu.<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
start = time.time()
outputs = trainer.model.generate(
    **inputs, 
    max_new_tokens=40,
    temperature=0.3,
    pad_token_id=tokenizer.eos_token_id
)
end = time.time()

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
split_str = "<|im_start|>assistant\n"
if split_str in response:
    result = response.split(split_str)[-1].replace("<|im_end|>", "").strip()
else:
    result = response

print(f"\n\n[INFERENCE SECONDS: {round(end - start, 2)}s]")
print(f"\n\n--- FINAL AI MANGLISH OUTPUT RESPONSE ---\n\n{result}\n\n--------------------------------------------\n")
