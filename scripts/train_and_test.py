import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

config = {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset_path": "data/manglish_booking_data.jsonl",
    "output_dir": "checkpoints/manglish-qwen",
    "learning_rate": 0.0008, 
    "num_train_epochs": 10, 
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "optim": "adamw_torch"
}

def run():
    print(f"Loading Base CPU: {config['model_id']} ...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        device_map="cpu", 
        torch_dtype=torch.float32 
    )
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    
    # Qwen uses generic pad tokens, but ChatML enforces structure.
    tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    dataset = load_dataset("json", data_files={"train": config["dataset_path"]}, split="train")
    dataset = dataset.select(range(min(100, len(dataset))))  # Use 100 samples to establish memory over CPU
    
    training_args = SFTConfig(
        dataset_text_field="text",
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=5,
        optim=config["optim"],
        num_train_epochs=config["num_train_epochs"],
        report_to="none",
        use_cpu=True
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("Executing backprop over ChatML tokens...")
    trainer.train()
    
    print("\n=======================================================")
    print("TRAINING DONE! RUNNING LIVE INFERENCE TEST!")
    print("=======================================================\n")
    
    # Notice we format the inference prompt using EXACTLY the ChatML format!
    prompt = "<|im_start|>system\nYou are StrawCore AI, a highly capable assistant specializing in handling client bookings, consultations, and payments. You understand English, phonetic Manglish, and Malayalam natively.<|im_end|>\n<|im_start|>user\nEnikku kochi yil oru hotel room book cheyyanam, 2 perundu.<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = trainer.model.generate(
        **inputs, 
        max_new_tokens=25, 
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    split_str = "<|im_start|>assistant\n"
    if split_str in response:
        result = response.split(split_str)[-1]
    else:
        result = response
        
    print(f"\n\n--- CLEAN ChatML MODEL OUTPUT MANGLISH/MALAYALAM TEST ---\n\n{result}\n\n--------------------------------------------\n")

if __name__ == "__main__":
    run()
