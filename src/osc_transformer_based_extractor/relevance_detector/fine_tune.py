"""
Fine-tune a Hugging Face transformer model on a custom dataset using CSV input.

This script performs the following steps:
1. Validates the input CSV file and output directory.
2. Loads the dataset and splits it into training and evaluation sets.
3. Fine-tunes a pre-trained Hugging Face transformer model on the dataset.
4. Evaluates the model and prints evaluation results and accuracy.
5. Saves the fine-tuned model and tokenizer.
"""

import os
import pandas as pd
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset


def check_csv_columns(file_path):
    """
    Check if the CSV file exists and contains the required columns.

    Args:
        file_path (str): Path to the CSV file.

    Raises:
        ValueError: If the file does not exist or does not contain the required columns.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Data path {file_path} does not exist.")

    required_columns = ["question", "context", "label"]

    try:
        df = pd.read_csv(file_path)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"CSV file must contain the columns: {required_columns}. Missing columns: {missing_columns}"
            )
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file.")
    except Exception as e:
        raise ValueError(f"Error while checking CSV columns: {e}")


def check_output_dir(output_dir):
    """
    Check if the output directory is valid.

    Args:
        output_dir (str): Path to the output directory.

    Raises:
        ValueError: If the directory does not exist or is not a directory.
    """
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output path {output_dir} is not a directory.")


class CustomDataset(Dataset):
    """
    Custom dataset class for sequence classification tasks.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding the data.
        questions (pd.Series): A pandas series containing the questions.
        contexts (pd.Series): A pandas series containing the contexts.
        labels (pd.Series): A pandas series containing the labels.
        max_length (int): The maximum length of the input sequences.
    """

    def __init__(self, tokenizer, questions, contexts, labels, max_length):
        """
        Initialize the CustomDataset.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding the data.
            questions (pd.Series): A pandas series containing the questions.
            contexts (pd.Series): A pandas series containing the contexts.
            labels (pd.Series): A pandas series containing the labels.
            max_length (int): The maximum length of the input sequences.
        """
        self.tokenizer = tokenizer
        self.questions = questions
        self.contexts = contexts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.questions)

    def __getitem__(self, idx):
        """
        Get the sample at index `idx`.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input IDs, attention mask, and labels.
        """
        question = str(self.questions[idx])
        context = str(self.contexts[idx])
        label = self.labels[idx]

        inputs = self.tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def fine_tune_model(
    data_path,
    model_name,
    num_labels,
    max_length,
    epochs,
    batch_size,
    output_dir,
    save_steps,
):
    """
    Fine-tune a pre-trained model on a custom dataset.

    Args:
        data_path (str): Path to the CSV file containing the dataset.
        model_name (str): Name/path of the pre-trained model to use.
        num_labels (int): Number of labels for the classification task.
        max_length (int): Maximum length of the input sequences.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        output_dir (str): Directory where the model will be saved during training.
        save_steps (int): Number of steps before saving the model during training.
    """
    # Load your dataset into a pandas DataFrame
    df = pd.read_csv(data_path)
    df = df[["question", "context", "label"]]

    # Load Model and Tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split the data into training and evaluation sets
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    # Create training and evaluation datasets
    train_dataset = CustomDataset(
        tokenizer,
        train_df["question"],
        train_df["context"],
        train_df["label"],
        max_length,
    )
    eval_dataset = CustomDataset(
        tokenizer, eval_df["question"], eval_df["context"], eval_df["label"], max_length
    )

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        logging_dir="./logs",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start Training
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate(eval_dataset)
    print("Evaluation results:")
    for key, value in eval_result.items():
        print(f"{key}: {value}")

    # Predict labels for the evaluation dataset
    predictions = trainer.predict(eval_dataset)
    predicted_labels = predictions.predictions.argmax(axis=1)
    true_labels = [
        eval_dataset[idx]["labels"].item() for idx in range(len(eval_dataset))
    ]

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)

    # Print inputs along with predicted labels
    for i in range(len(eval_dataset)):
        eva_data = eval_dataset[i]
        input_ids = eva_data["input_ids"]
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        print(f"Input: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
        print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
        print()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
