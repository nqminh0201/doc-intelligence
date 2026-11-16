"""
evaluate.py — Evaluate the Fine-Tuned LoRA Adapter vs. the Base Model.

WHAT THIS FILE DOES:
    1. Loads the validation set (`training/data/val.jsonl`).
    2. Computes Cross-Entropy Loss and Perplexity on the Base Model vs. the
       Fine-Tuned LoRA Model.
    3. Generates side-by-side answers for validation questions so you can visually
       inspect how LoRA improved citation formatting and factual grounding.

KEY CONCEPTS:
    - Perplexity (PPL = exp(Loss)):
        Measures how "uncertain" or "perplexed" a language model is when predicting
        the next token on unseen validation data.
        * Lower Perplexity = the model is more confident and accurate on your domain.
    - Adapter Merging / PeftModel:
        At inference time, we load the frozen Base Model and attach our tiny
        `lora_adapter` folder (~2 MB) on top of it using `PeftModel.from_pretrained()`.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from train import BASE_MODEL_NAME, ADAPTER_SAVE_DIR, DATA_DIR, MAX_SEQ_LENGTH


def compute_loss(model: torch.nn.Module, tokenizer: AutoTokenizer, records: List[Dict[str, str]], device: str) -> float:
    """Compute average Cross-Entropy loss across a list of evaluation records."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for rec in records:
            text = rec["text"] + tokenizer.eos_token
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(device)
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=encoded["input_ids"],
            )
            total_loss += outputs.loss.item()
    return total_loss / max(len(records), 1)


def generate_answer(model: torch.nn.Module, tokenizer: AutoTokenizer, record: Dict[str, str], device: str) -> str:
    """Generate a response for a single validation question."""
    prompt = (
        "### Instruction:\n"
        "You are an expert Document Intelligence Assistant. Answer the question accurately "
        "using ONLY the provided context, and always cite the source.\n\n"
        f"### Context:\n{record['context']}\n\n"
        f"### Question:\n{record['question']}\n\n"
        "### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            min_new_tokens=20,
            max_new_tokens=60,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens after the prompt
    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def evaluate_models() -> None:
    print("=" * 60)
    print("PHASE 7: BASE MODEL VS. LORA ADAPTER EVALUATION")
    print("=" * 60)

    val_path = DATA_DIR / "val.jsonl"
    if not val_path.exists() or not ADAPTER_SAVE_DIR.exists():
        print("[-] Validation dataset or trained LoRA adapter not found.")
        print("    Please run `python training/prepare_dataset.py` and `python training/train.py` first.")
        return

    val_records = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Evaluate Base Model
    print("[1/3] Evaluating Untrained Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, torch_dtype=torch.float32).to(device)
    base_loss = compute_loss(base_model, tokenizer, val_records, device)
    base_ppl = math.exp(base_loss)
    sample_base_ans = generate_answer(base_model, tokenizer, val_records[0], device)

    # 2. Load & Evaluate Fine-Tuned LoRA Model
    print("[2/3] Loading Fine-Tuned LoRA Adapter & Evaluating...")
    lora_model = PeftModel.from_pretrained(base_model, ADAPTER_SAVE_DIR).to(device)
    lora_loss = compute_loss(lora_model, tokenizer, val_records, device)
    lora_ppl = math.exp(lora_loss)
    sample_lora_ans = generate_answer(lora_model, tokenizer, val_records[0], device)

    # 3. Print Comparison Table
    print("\n" + "=" * 60)
    print("QUANTITATIVE METRICS (VALIDATION SET)")
    print("=" * 60)
    print(f"  Base Model Loss       : {base_loss:.4f}  (Perplexity: {base_ppl:.2f})")
    print(f"  Fine-Tuned LoRA Loss  : {lora_loss:.4f}  (Perplexity: {lora_ppl:.2f})")
    improvement = ((base_loss - lora_loss) / base_loss) * 100
    print(f"  Loss Reduction        : {improvement:.1f}% improvement!")

    print("\n" + "=" * 60)
    print("QUALITATIVE SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print(f"Question:\n  {val_records[0]['question']}\n")
    print(f"[Base Model Output]:\n  {sample_base_ans}\n")
    print(f"[Fine-Tuned LoRA Output]:\n  {sample_lora_ans}\n")
    print(f"[Ground Truth Target]:\n  {val_records[0]['answer']}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_models()
