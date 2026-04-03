import yaml
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

def run_3b_sft():
    with open("configs/qlora_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print(f"LOADING 3B MODEL (BF16 CPU): {config['model_id']}")
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print(f"Dataset Loading: {config['dataset_path']}...")
    dataset = load_dataset("json", data_files={"train": config["dataset_path"]}, split="train")
    dataset = dataset.select(range(min(500, len(dataset))))
    
    # In newer TRL versions, max_seq_length is sometimes part of SFTConfig or SFTTrainer
    training_args = SFTConfig(
        dataset_text_field="text",
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        logging_steps=config["logging_steps"],
        optim=config["optim"],
        num_train_epochs=config["num_train_epochs"],
        report_to="none",
        use_cpu=True,
        gradient_checkpointing=config["gradient_checkpointing"]
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # Explicitly set max_seq_length on trainer if needed
    if hasattr(config, 'max_seq_length'):
        trainer.max_seq_length = config['max_seq_length']

    print("LAUNCHING 3B MANG-TRAIN (EPYC CPU + NVMe SWAP)...")
    trainer.train()
    
    print("SUCCESS! Saving 3B Adapters...")
    trainer.model.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    run_3b_sft()
