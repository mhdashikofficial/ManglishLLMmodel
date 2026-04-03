import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("Phase 1: Loading Llama-3.2-1B-Instruct...")
model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_id = "checkpoints/manglish-llama-1b"

# Load without gated access check if cached, or prompt token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("Phase 2: Loading dataset...")
dataset = load_dataset("json", data_files={"train": "data/manglish_booking_data.jsonl"}, split="train")
dataset = dataset.select(range(min(150, len(dataset)))) 

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

trainer = SFTTrainer(
    model=model, train_dataset=dataset, processing_class=tokenizer, args=training_args
)

print("Phase 3: Training on EPYC CPU (5 Epochs)...")
trainer.train()

print(f"Phase 4: Saving Weights...")
trainer.model.save_pretrained(adapter_id)

print("\n--- INFERENCE TEST ---")
prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are StrawCore AI, specializing in bookings. You understand English, Manglish, and Malayalam natively.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nEnikku kochi yil oru hotel room book cheyyanam, 2 perundu.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
outputs = trainer.model.generate(**inputs, max_new_tokens=40, temperature=0.3, pad_token_id=tokenizer.eos_token_id)
print(f"OUTPUT: {tokenizer.decode(outputs[0], skip_special_tokens=True).split('assistant')[-1].strip()}")
