# All imports at the top
import torch
import os
import shutil
import numpy as np
import pandas as pd
import mlflow
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

os.environ["HF_TOKEN"] = "your_huggingface_token"

# Configuration
model_name = "mistralai/Mathstral-7B-v0.1"
MAX_LEN = 256

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:8081")

# Step 2: Loading the dataset

le = LabelEncoder()
train = pd.read_csv('train.csv')
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category +":"+ train.Misconception
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_)
print(f"Train shape: {train.shape} with {n_classes} target classes")
print(train.head())

# Process correct answers
idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

# Format input text
def format_input(row):
    x = "This answer is correct."
    if not row['is_correct']:
        x = "This is answer is incorrect."
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"{x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input, axis=1)

# Split data
train_df          = train
#train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

COLS = ['text', 'label']
train_ds = Dataset.from_pandas(train_df[COLS])
#val_ds = Dataset.from_pandas(val_df[COLS])

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ["HF_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_func(example):
    return tokenizer(
        example["text"],
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )

# Tokenize datasets
train_ds = train_ds.map(tokenize_func, batched=True, desc="Tokenizing train data")
#eval_ds = val_ds.map(tokenize_func, batched=True, desc="Tokenizing eval data")

# Step 3: Load model
# Model configuration
model_kwargs = dict(
    trust_remote_code=True,
    torch_dtype=torch.float16
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16",
)

# Load model
print(f"Loading model : {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, use_cache=False, num_labels=n_classes, token=os.environ["HF_TOKEN"], **model_kwargs
)

model.config.pad_token_id = tokenizer.pad_token_id

# LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score"],
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Custom evaluation metric
def compute_multi_map(eval_pred, ks=[3, 5, 10]):
    """
    Computes MAP@k and a detailed rank distribution.
    
    This includes:
    - Rank counts for rank 1, 2-3, and above 3.
    - For rank groups 2-3 and above 3, it finds the top 3 most frequent
      classes and calculates their average probability score.
    """
    # 1. Unpack logits and labels
    logits, labels = eval_pred
    labels = np.array(labels)

    # 2. Convert logits to probabilities
    # The `probs` array has shape: (num_samples, num_classes)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # 3. Get top-k predictions
    max_k = max(ks)
    top_k_preds = np.argsort(-probs, axis=1)[:, :max_k]

    # 4. Create a boolean match array
    match_array = (top_k_preds == labels[:, None])

    # 5. Compute MAP@k for each specified k
    metrics = {}
    for k in ks:
        match_at_k = match_array[:, :k]
        ranks = np.argmax(match_at_k, axis=1) + 1
        has_match_at_k = np.any(match_at_k, axis=1)
        scores = has_match_at_k * (1.0 / ranks)
        metrics[f"map@{k}"] = np.mean(scores)

    # 6. Calculate detailed rank position breakdown
    ranks_with_indices = [np.where(row)[0] for row in match_array]
    correct_ranks = np.array([r[0] + 1 if len(r) > 0 else max_k + 1 for r in ranks_with_indices])

    total = labels.shape[0]
    metrics["rank_1"] = np.sum(correct_ranks == 1)
    metrics["rank_2_to_3"] = np.sum((correct_ranks >= 2) & (correct_ranks <= 3))
    metrics["rank_above_3"] = np.sum((correct_ranks > 3) & (correct_ranks <= max_k))
    metrics["no_match_in_top_k"] = np.sum(correct_ranks > max_k)
    metrics["total"] = total

    # 7. Find top 3 classes for rank groups and their average probability
    
    # --- For ranks 2 to 3 ---
    # Create a boolean mask for samples in this rank group
    rank_2_to_3_mask = (correct_ranks >= 2) & (correct_ranks <= 3)
    # Get the true labels for these samples
    rank_2_to_3_labels = labels[rank_2_to_3_mask]

    if len(rank_2_to_3_labels) > 0:
        top_classes = Counter(rank_2_to_3_labels).most_common(3)
        augmented_top_classes = []
        for cls, count in top_classes:
            # Find samples that both belong to this class AND are in this rank group
            class_in_group_mask = (labels == cls) & rank_2_to_3_mask
            # Get the probabilities assigned to the correct class for these specific samples
            class_probs = probs[class_in_group_mask, cls]
            # Calculate the average probability and add to list
            avg_prob = np.mean(class_probs)
            augmented_top_classes.append((cls, count, round(float(avg_prob), 4)))
        metrics["rank_2_to_3_details"] = augmented_top_classes
    else:
        metrics["rank_2_to_3_details"] = []

    # --- For ranks above 3 (up to max_k) ---
    rank_above_3_mask = (correct_ranks > 3) & (correct_ranks <= max_k)
    rank_above_3_labels = labels[rank_above_3_mask]

    if len(rank_above_3_labels) > 0:
        top_classes = Counter(rank_above_3_labels).most_common(3)
        augmented_top_classes = []
        for cls, count in top_classes:
            class_in_group_mask = (labels == cls) & rank_above_3_mask
            class_probs = probs[class_in_group_mask, cls]
            avg_prob = np.mean(class_probs)
            augmented_top_classes.append((cls, count, round(float(avg_prob), 4)))
        metrics["rank_above_3_details"] = augmented_top_classes
    else:
        metrics["rank_above_3_details"] = []

    mlflow.log_metric("rank_1", metrics["rank_1"])
    mlflow.log_metric("rank_2_to_3", metrics["rank_2_to_3"])
    mlflow.log_metric("rank_above_3", metrics["rank_above_3"])
    mlflow.log_metric("no_match_in_top_k", metrics["no_match_in_top_k"])
    # mlflow.log_metric("rank_2_to_3_details", metrics["rank_2_to_3_details"])
    # mlflow.log_metric("rank_above_3_details", metrics["rank_above_3_details"])

    return metrics

# Training arguments
training_args = TrainingArguments(
    output_dir="MAP_EXP_14_FULL",
    eval_strategy="no",
    save_strategy="no",
    logging_strategy="steps",
    #eval_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="mlflow",
    gradient_checkpointing=True,
    group_by_length=True,
    max_grad_norm=1.0,
    weight_decay=0.01,
    num_train_epochs=2
)


import torch
import numpy as np
import mlflow
from collections import Counter
from transformers import Trainer

class MLflowMetricsLogger:
    """
    A callable class to compute and log metrics to MLflow with step tracking.
    """
    def __init__(self, trainer: Trainer, ks=[3, 5, 10]):
        """
        Initializes the metrics logger.
        
        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            ks (list): A list of k values for MAP@k calculation.
        """
        self.trainer = trainer
        self.ks = ks

    def __call__(self, eval_pred):
        """
        This method is called by the Trainer during evaluation.
        """
        # Get the current training step from the trainer's state
        step = self.trainer.state.global_step

        # 1. Unpack logits and labels
        logits, labels = eval_pred
        labels = np.array(labels)

        # 2. Convert logits to probabilities
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

        # 3. Get top-k predictions
        max_k = max(self.ks)
        top_k_preds = np.argsort(-probs, axis=1)[:, :max_k]

        # 4. Create a boolean match array
        match_array = (top_k_preds == labels[:, None])

        # 5. Compute MAP@k for each specified k
        metrics = {}
        for k in self.ks:
            match_at_k = match_array[:, :k]
            ranks = np.argmax(match_at_k, axis=1) + 1
            has_match_at_k = np.any(match_at_k, axis=1)
            scores = has_match_at_k * (1.0 / ranks)
            metrics[f"map@{k}"] = np.mean(scores)

        # 6. Calculate detailed rank position breakdown
        ranks_with_indices = [np.where(row)[0] for row in match_array]
        correct_ranks = np.array([r[0] + 1 if len(r) > 0 else max_k + 1 for r in ranks_with_indices])

        total = labels.shape[0]
        rank_1_count = np.sum(correct_ranks == 1)
        rank_2_to_3_count = np.sum((correct_ranks >= 2) & (correct_ranks <= 3))
        rank_above_3_count = np.sum((correct_ranks > 3) & (correct_ranks <= max_k))
        no_match_count = np.sum(correct_ranks > max_k)

        # Log metrics to MLflow WITH the step argument
        mlflow.log_metric("rank_1", rank_1_count, step=step)
        mlflow.log_metric("rank_2_to_3", rank_2_to_3_count, step=step)
        mlflow.log_metric("rank_above_3", rank_above_3_count, step=step)
        mlflow.log_metric("no_match_in_top_k", no_match_count, step=step)

        # Note: The detailed lists cannot be logged as a time-series metric.
        # These are better logged as artifacts (e.g., a JSON file) or a dictionary
        # at the end of the run if needed.
        # For example: mlflow.log_dict(details_dict, "rank_details.json")

        # The Trainer still requires a dictionary of metrics to be returned.
        metrics["rank_1"] = rank_1_count
        metrics["rank_2_to_3"] = rank_2_to_3_count
        metrics["rank_above_3"] = rank_above_3_count
        metrics["no_match_in_top_k"] = no_match_count
        metrics["total"] = total
        
        return metrics


# Initialize trainer
trainer = Trainer(
    model,
    args=training_args,
    train_dataset=train_ds,
    #eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_multi_map,
    data_collator=DataCollatorWithPadding(tokenizer),
)

metrics_computer = MLflowMetricsLogger(trainer)

# 3. Assign the instance to the trainer's compute_metrics attribute
trainer.compute_metrics = metrics_computer

# Main execution
if __name__ == "__main__":
    
    # Start training
    trainer.train()
    
    # Save the model
    trainer.save_model("MAP_EXP_14_FULL")

    source_file = "MAP_EXP_14_FULL.py"
    destination_directory = "MAP_EXP_14_FULL"
    
    shutil.copy(source_file, destination_directory)
    print(f"File '{source_file}' copied to '{destination_directory}'")
    