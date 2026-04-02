<div align="center">
  <img src="https://strawhatai.space/assets/images/logo.png" alt="StrawCore AI Logo" width="200" />
  <h1>StrawCore AI: Manglish Base Model 🚀</h1>
  <p><i>Empowering specialized bot systems with a unified English, Malayalam, and Manglish semantic engine.</i></p>
</div>

---

## 📖 Overview

This repository holds the code, datasets, and configurations for fine-tuning the **StrawCore AI Manglish Language Model**. Built on top of powerful open-source base models like Llama 3 8B, this model is explicitly designed to handle domain-specific workflows (such as academic booking, consultations, hotel reservations, and medical booking) in mixed English, Manglish, and native Malayalam script.

## 🧠 Capabilities

- **Tri-Lingual Understanding:** Seamlessly parses instructions written in English, phonetic Manglish ("enikku oru hotel book cheyyanam"), and native Malayalam (എനിക്ക് ഒരു ഹോട്ടൽ ബുക്ക് ചെയ്യണം).
- **Domain Specialization:** Highly attuned to booking flows, data collection intent, and conversational logic.
- **Fast & Efficient:** Fine-tuned via QLoRA, making it optimal for local hosting via Ollama and local deployment architectures like OpenClaw.

## 🛠️ Tech Stack

- **Base Models:** Llama-3-8B-Instruct (or Mistral-7B-Instruct)
- **Training Libraries:** HuggingFace `transformers`, `peft` (LoRA/QLoRA), `trl`, and `bitsandbytes`
- **Deployment:** Ollama, vLLM
- **Agent Integration:** OpenClaw Agent Framework

## 🚀 Getting Started

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generating the Dataset
To train the model on booking workflows, use the synthetic data generator:
```bash
python scripts/generate_synthetic_data.py
```

### 3. Fine-Tuning the Model
Run the QLoRA fine-tuning script. The parameters are defined in `configs/qlora_config.yaml`.
```bash
python scripts/train_qlora.py
```

### 4. Integration with Ollama
Once fine-tuned, export the model to GGUF format and load it using the provided Modelfile:
```bash
ollama create strawcore-manglish -f integrations/ollama_Modelfile
ollama run strawcore-manglish
```

---
*Created for StrawCore AI*
