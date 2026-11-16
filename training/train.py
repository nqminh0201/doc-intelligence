"""
train.py — Fine-Tune a Language Model using LoRA (Low-Rank Adaptation).

WHAT THIS FILE DOES:
    1. Loads the instruction-tuning dataset created by `prepare_dataset.py`.
    2. Loads a base Causal Language Model (`HuggingFaceTB/SmolLM2-135M-Instruct`,
       which shares the exact Llama-style Transformer architecture as Llama 3.2,
       sized to train in under a minute on a standard CPU).
    3. Attaches LoRA (Low-Rank Adaptation) matrices to the attention layers (`q_proj`, `v_proj`),
       freezing ~99% of the original weights so only a tiny adapter (~0.3% of weights) is trained.
    4. Runs a PyTorch training loop with gradient accumulation and AdamW optimizer,
       logging loss at each epoch.
    5. Saves the trained LoRA adapter weights to `training/lora_adapter/`.

KEY CONCEPTS:
    - Why LoRA instead of Full Fine-Tuning?
        Full fine-tuning updates EVERY weight in a model (billions of numbers), requiring
        massive GPU clusters. LoRA freezes the original weight matrix W (size d x d) and
        injects two tiny matrices A (size d x r) and B (size r x d), where rank r << d
        (e.g., r = 8).
        Math: Output = W*x + (alpha / r) * (B * A * x)
        Instead of training 4,096 x 4,096 = 16.7M parameters per layer, LoRA trains
        2 x (4,096 x 8) = 65K parameters — a 99.6% reduction!

    - Causal Language Modeling Loss (Cross-Entropy):
        The model predicts the probability distribution of the NEXT token given all
        previous tokens. Loss measures how surprised the model is by the true next token.
        Lower loss = the model has learned the domain style and knowledge.
"""

import json
import os
from pathlib import Path
from typing import List, Dict

# Disable hf-xet & symlink warnings on Windows so Hugging Face downloads cleanly over HTTPS
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# ── Paths & Hyperparameters ────────────────────────────────────────
TRAINING_DIR = Path(__file__).parent
DATA_DIR = TRAINING_DIR / "data"
ADAPTER_SAVE_DIR = TRAINING_DIR / "lora_adapter"

# SmolLM2-135M-Instruct uses the Llama Transformer architecture (q_proj, v_proj)
# and runs comfortably on CPU while demonstrating real production LoRA.
BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

LORA_RANK = 8            # Rank (r) of the low-rank decomposition matrices A and B
LORA_ALPHA = 16          # Scaling factor (alpha / r scales the adapter output)
LORA_DROPOUT = 0.05      # Dropout probability for regularization
LEARNING_RATE = 3e-4     # Standard LoRA learning rate (higher than full fine-tuning)
EPOCHS = 5               # Number of complete passes over the training dataset
MAX_SEQ_LENGTH = 256     # Maximum token length per training example
BATCH_SIZE = 2           # Examples processed per forward pass


class InstructionDataset(Dataset):
    """
    PyTorch Dataset that tokenizes instruction-response texts into
    `input_ids`, `attention_mask`, and `labels` for Causal LM training.
    """

    def __init__(self, jsonl_path: Path, tokenizer: AutoTokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.records: List[Dict[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.records[idx]["text"] + self.tokenizer.eos_token
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # In Causal LM training, labels are identical to input_ids (shifted internally by the model).
        # We mask padding tokens with -100 so PyTorch CrossEntropyLoss ignores padding!
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_lora() -> None:
    print("=" * 60)
    print("PHASE 7: LORA FINE-TUNING PIPELINE")
    print("=" * 60)

    train_jsonl = DATA_DIR / "train.jsonl"
    if not train_jsonl.exists():
        print("[~] Dataset not found. Running prepare_dataset.py first...")
        from prepare_dataset import prepare_datasets
        prepare_datasets()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Compute Device : {device.upper()}")
    print(f"[+] Base Model     : {BASE_MODEL_NAME}")

    # 1. Load Tokenizer & Base Model
    print("\n[1/4] Loading tokenizer and pre-trained base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float32,
    )

    # 2. Configure & Attach LoRA Adapter
    print("\n[2/4] Injecting LoRA low-rank adapter matrices into Attention layers...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],  # Query & Value projection matrices in Self-Attention
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(device)

    # Print trainable parameter reduction!
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    pct = 100 * trainable_params / all_params
    print(f"    Total Parameters     : {all_params:,}")
    print(f"    Trainable (LoRA)     : {trainable_params:,} ({pct:.3f}% of total!)")
    print(f"    Frozen Parameters    : {all_params - trainable_params:,}")

    # 3. Prepare DataLoader & Optimizer
    print("\n[3/4] Starting LoRA training loop...")
    train_dataset = InstructionDataset(train_jsonl, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=LEARNING_RATE)

    peft_model.train()
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = peft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"    Epoch {epoch}/{EPOCHS} — Average Cross-Entropy Loss: {avg_loss:.4f}")

    # 4. Save the LoRA Adapter Weights
    print("\n[4/4] Saving trained LoRA adapter...")
    ADAPTER_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(ADAPTER_SAVE_DIR)
    tokenizer.save_pretrained(ADAPTER_SAVE_DIR)

    # Save training metrics summary
    metrics_path = ADAPTER_SAVE_DIR / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": BASE_MODEL_NAME,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "epochs": EPOCHS,
                "initial_loss": round(loss_history[0], 4),
                "final_loss": round(loss_history[-1], 4),
                "loss_history": [round(x, 4) for x in loss_history],
                "trainable_parameters": trainable_params,
                "total_parameters": all_params,
            },
            f,
            indent=2,
        )

    print(f"[+] LoRA adapter saved to : {ADAPTER_SAVE_DIR}")
    print(f"[+] Loss decreased from   : {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train_lora()
