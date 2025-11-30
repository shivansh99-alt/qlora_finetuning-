# QLoRA GPT-2 Finetuning (4-bit + LoRA)

A compact, reproducible example showing how to fine-tune **GPT-2** using QLoRA (bitsandbytes 4-bit quantization + LoRA adapters).  
This repo contains a Colab-style script/notebook that loads a dataset, tokenizes, applies 4-bit quantization, prepares the model for k-bit training, attaches a LoRA adapter, trains, and saves the adapter for later inference. :contentReference[oaicite:1]{index=1}

---

## Highlights
- Load a text dataset and tokenize for causal LM training.
- Use `bitsandbytes` 4-bit quantization for small memory footprint.
- Prepare model using `peft` / `prepare_model_for_kbit_training`.
- Add LoRA adapter for parameter-efficient fine-tuning (save only LoRA).
- Example of loading the base model + LoRA for inference.

---

## Files
- `gpt_2_finetuning_qlora.py` — main Colab-style script/notebook (training + saving adapter). :contentReference[oaicite:2]{index=2}

---

## Requirements
Recommended to run in a GPU environment (Colab with A100/16GB+ or similar).

```bash
pip install transformers datasets peft accelerate bitsandbytes torch
