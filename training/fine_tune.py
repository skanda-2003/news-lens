"""
Fine-tune roberta-base on the Qbias AllSides dataset for 3-class bias classification.

Run from the project root:
    python training/fine_tune.py

Reads:  data/processed/train.csv, data/processed/val.csv
Writes: models/bias_classifier/
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm


# ── hyperparameters ──────────────────────────────────────────────────────────

MODEL_NAME       = "roberta-base"
TRAIN_PATH       = "data/processed/train.csv"
VAL_PATH         = "data/processed/val.csv"
MODEL_OUTPUT_DIR = "models/bias_classifier"

BATCH_SIZE       = 16   # articles per GPU step
GRAD_ACCUM_STEPS = 2    # step the optimiser every 2 batches → effective batch size = 32
LEARNING_RATE    = 2e-5
NUM_EPOCHS       = 3
MAX_LENGTH       = 512  # RoBERTa's hard token limit

# must match the label mapping used in 02_classifier_eval.ipynb
LABEL2ID = {"left": 0, "center": 1, "right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── dataset class ─────────────────────────────────────────────────────────────

class BiasDataset(Dataset):
    """
    Wraps a DataFrame so PyTorch's DataLoader can pull batches from it.

    PyTorch's DataLoader expects a Dataset object with __len__ and __getitem__.
    __len__ tells it how many samples exist; __getitem__ tells it how to fetch one.
    """

    def __init__(self, df, tokenizer):
        # reset_index so iloc[0] always means the first row regardless of the original index
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # build the input string: headline + RoBERTa separator token + first 400 chars of body
        # </s> is RoBERTa's separator - not [SEP], which is BERT-specific
        text = f"{row['heading']} </s> {str(row['text'])[:400]}"

        # tokenize: convert text to token IDs the model understands
        # truncation=True: if text is longer than MAX_LENGTH tokens, cut it off
        # padding is done per-batch by DataCollatorWithPadding, not here
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # attach the integer label so the model can compute cross-entropy loss
        encoded["labels"] = int(row["label"])
        return encoded


# ── evaluation function ───────────────────────────────────────────────────────

def evaluate(model, loader, device):
    """
    Run inference on a DataLoader. Returns (avg_loss, macro_f1).

    I call this after every epoch to decide whether to save a new checkpoint.
    """
    model.eval()  # disables dropout - must do this before any evaluation pass

    all_preds  = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():  # skip gradient tracking - faster and uses less memory
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)  # passing labels= makes the model compute loss internally

            total_loss += output.loss.item()

            # argmax picks the class index with the highest score
            preds = output.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / len(loader)

    # average="macro" gives equal weight to each class regardless of sample count
    # this is the right metric when class sizes are unequal (left >> center in our data)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, macro_f1


# ── training ──────────────────────────────────────────────────────────────────

def train():
    print(f"Device: {DEVICE}")

    # load the pre-split CSVs from the notebook
    print(f"\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    print(f"  Train: {len(train_df):,} articles")
    print(f"  Val:   {len(val_df):,} articles")

    # the tokenizer converts raw text to integers (token IDs) that roberta-base understands
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # DataCollatorWithPadding pads each batch to its longest sequence
    # more efficient than padding every sequence to 512 globally
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = BiasDataset(train_df, tokenizer)
    val_dataset   = BiasDataset(val_df,   tokenizer)

    # shuffle=True for training so the model doesn't memorise batch order
    # shuffle=False for val so results are deterministic
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    # load roberta-base with a fresh 3-class classification head on top
    # the head is a single linear layer: 768-dim RoBERTa output → 3 class scores
    print(f"\nLoading model: {MODEL_NAME} (num_labels=3)")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = model.to(DEVICE)

    # AdamW is the standard optimiser for transformer fine-tuning
    # weight_decay adds L2 regularisation - penalises large weights to reduce overfitting
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # total number of optimiser steps over all epochs
    # we only call optimizer.step() every GRAD_ACCUM_STEPS batches, hence the division
    total_steps  = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * 0.1)  # warm up for the first 10% of training

    # linear warmup then linear decay: LR ramps 0 → LEARNING_RATE over warmup_steps,
    # then decays LEARNING_RATE → 0 over the remaining steps
    # warmup prevents instability from large gradient updates right at the start
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nHyperparameters:")
    print(f"  Epochs:               {NUM_EPOCHS}")
    print(f"  Batch size:           {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"  Learning rate:        {LEARNING_RATE}")
    print(f"  Total optimiser steps:{total_steps}")
    print(f"  Warmup steps:         {warmup_steps}")
    print(f"\nStarting training...\n")

    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()  # re-enable dropout for the training pass
        total_train_loss = 0.0
        optimizer.zero_grad()

        # tqdm adds a progress bar so I can see how far through each epoch I am
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            unit="batch",
        )

        for step, batch in enumerate(progress):
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            output = model(**batch)

            # divide loss before calling backward so accumulated gradients are
            # equivalent to computing the gradient on the full effective batch at once
            loss = output.loss / GRAD_ACCUM_STEPS
            loss.backward()

            total_train_loss += output.loss.item()

            # only update weights after accumulating GRAD_ACCUM_STEPS gradients
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # clip gradients to norm 1.0 - prevents exploding gradients,
                # a common instability when fine-tuning large transformers
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # show the per-step loss in the progress bar
            progress.set_postfix({"loss": f"{output.loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # evaluate on the validation set after every epoch
        val_loss, val_f1 = evaluate(model, val_loader, DEVICE)

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
            f" | Train loss: {avg_train_loss:.4f}"
            f" | Val loss: {val_loss:.4f}"
            f" | Val macro F1: {val_f1:.4f}"
        )

        # save checkpoint only if this epoch produced the best val macro F1 so far
        # I save based on F1, not loss - a lower loss doesn't always mean better classification
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
            print(f"  -> New best checkpoint saved  (val macro F1: {val_f1:.4f})")
        else:
            print(f"  -> No improvement             (best so far:  {best_val_f1:.4f})")

        print()

    print(f"Training complete.")
    print(f"Best val macro F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to: {MODEL_OUTPUT_DIR}/")


if __name__ == "__main__":
    train()
