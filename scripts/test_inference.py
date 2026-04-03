import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_model_id = "checkpoints/manglish-qwen"

print("Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="cpu", 
    torch_dtype=torch.float32 
)

print("Applying fine-tuned Manglish LoRA adapters...")
try:
    model = PeftModel.from_pretrained(base_model, adapter_model_id)
except Exception as e:
    print(f"ERROR loading adapters: {e}. Was training fully completed?")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

prompt = "<|im_start|>system\nYou are StrawCore AI, a highly capable assistant specializing in handling client bookings, consultations, and payments. You understand English, phonetic Manglish, and Malayalam natively.<|im_end|>\n<|im_start|>user\nEnikku kochi yil oru hotel room book cheyyanam, 2 perundu.<|im_end|>\n<|im_start|>assistant\n"

print("\n-----------------------------------")
print("Executing generation on CPU (please wait)...")

inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
start = time.time()
outputs = model.generate(
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

print(f"\n\n[INFERENCE TIME: {round(end - start, 2)}s]")
print(f"\n\n--- MODEL OUTPUT MANGLISH/MALAYALAM TEST ---\n\n{result}\n\n--------------------------------------------\n")
