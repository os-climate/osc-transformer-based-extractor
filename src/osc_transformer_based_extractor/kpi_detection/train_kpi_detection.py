"""
KPI Detection Training Module

This module provides functionality to fine-tune a pre-trained kpi-detection model on a custom dataset.
The training process involves loading the data, preprocessing it, and training the model using the Hugging Face
Transformers library.

Functions:
    train_model(data_path, model_name, max_length, epochs, batch_size, output_dir, save_steps):
        Fine-tunes a pre-trained model on a custom dataset and evaluates its performance.

Example usage:
    train_model(
        data_path='path/to/your/dataset.csv',
        model_name='bert-base-uncased',
        max_length=512,
        epochs=3,
        batch_size=8,
        output_dir='./model_output',
        save_steps=500,
    )
"""

import pandas as pd
from functools import partial
import os
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DefaultDataCollator,
)
import numpy as np
from sklearn.model_selection import train_test_split


def check_csv_columns_kpi_detection(file_path):
    """
    Check if the CSV file exists and contains the required columns.

    Args:
        file_path (str): Path to the CSV file.

    Raises:
        ValueError: If the file does not exist or does not contain the required columns.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Data path {file_path} does not exist.")

    required_columns = ["question", "context", "annotation_answer"]

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


def train_kpi_detection(
    data_path,
    model_name,
    max_length,
    epochs,
    batch_size,
    learning_rate,
    output_dir,
    save_steps,
):
    """
    Fine-tune a pre-trained model on a custom dataset.

    Args:
        data_path (str): Path to the CSV file containing the dataset.
        model_name (str): Name/path of the pre-trained model to use.
        max_length (int): Maximum length of the input sequences.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for trainig
        output_dir (str): Directory where the model will be saved during training.
        save_steps (int): Number of steps before saving the model during training.
    """
    # Load the data
    df = pd.read_csv(data_path)
    df["annotation_answer"] = df["annotation_answer"].astype(str)
    df = df[["question", "context", "annotation_answer", "answer_start"]]

    def expand_rows(df, column):
        # Create a new DataFrame where each list element becomes a separate row
        rows = []
        for _, row in df.iterrows():
            if isinstance(row[column], list):
                for value in row[column]:
                    new_row = row.copy()
                    new_row[column] = value
                    rows.append(new_row)
            else:
                rows.append(row)

        # Convert the list of rows back to a DataFrame
        return pd.DataFrame(rows)

    # Apply the function to the DataFrame
    new_df = expand_rows(df, "answer_start")

    # Split the DataFrame into train and test sets
    train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Create a DatasetDict
    data = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def preprocess_function(examples, max_length):
        questions = examples["question"]
        contexts = examples["context"]
        answers = examples["annotation_answer"]
        answer_starts = examples["answer_start"]

        # Tokenize questions and contexts
        tokenized_inputs = tokenizer(
            questions,
            contexts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # Initialize lists to hold start and end positions
        start_positions = []
        end_positions = []

        # Loop through each example
        for i in range(len(questions)):
            # Get the answer start index
            answer_start = answer_starts[i]
            answer = answers[i]

            if answer_start == -1:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_positions.append(
                    tokenizer.encode(
                        contexts[i][:answer_start], add_special_tokens=False
                    ).__len__()
                )
                end_positions.append(
                    tokenizer.encode(
                        contexts[i][: answer_start + len(answer)],
                        add_special_tokens=False,
                    ).__len__()
                    - 1
                )

        tokenized_inputs.update(
            {"start_positions": start_positions, "end_positions": end_positions}
        )

        return tokenized_inputs

    # Apply the preprocessing function to the dataset
    # Create a new function with max_length set using partial
    preprocess_function_with_max_length = partial(
        preprocess_function, max_length=max_length
    )

    # Apply the preprocessing function to the dataset
    processed_datasets = data.map(preprocess_function_with_max_length, batched=True)

    # Remove columns that are not needed
    processed_datasets = processed_datasets.remove_columns(
        ["question", "context", "annotation_answer", "answer_start"]
    )

    data_collator = DefaultDataCollator()
    saved_model_path = os.path.join(output_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=saved_model_path,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        logging_dir="./logs",  # Directory for logs
        logging_steps=10,  # Log every 10 steps
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=save_steps,
        weight_decay=0.01,
        push_to_hub=False,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Evaluate
    eval_result = trainer.evaluate(processed_datasets["test"])
    print("Evaluation results:")
    for key, value in eval_result.items():
        print(f"{key}: {value}")

    # Predict labels for the evaluation dataset
    predictions = trainer.predict(processed_datasets["test"])
    start_logits = predictions.predictions[0]  # Start logits
    end_logits = predictions.predictions[1]  # End logits

    # Convert logits to start and end positions
    predicted_starts = np.argmax(start_logits, axis=1)
    predicted_ends = np.argmax(end_logits, axis=1)

    # Extract true start and end positions from the dataset
    true_starts = np.array(
        [example["start_positions"] for example in processed_datasets["test"]]
    )
    true_ends = np.array(
        [example["end_positions"] for example in processed_datasets["test"]]
    )

    # Calculate accuracy (you might want a different metric depending on your needs)
    accuracy = np.mean(
        (predicted_starts == true_starts) & (predicted_ends == true_ends)
    )
    print("Accuracy:", accuracy)

    # Print inputs along with predicted and true answer spans
    for i in range(len(processed_datasets["test"])):
        eva_data = processed_datasets["test"][i]
        input_ids = eva_data["input_ids"]
        true_start = true_starts[i]
        true_end = true_ends[i]
        predicted_start = predicted_starts[i]
        predicted_end = predicted_ends[i]

        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        predicted_answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                input_ids[predicted_start : predicted_end + 1]
            )
        )
        true_answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[true_start : true_end + 1])
        )

        print(f"Input: {input_text}")
        print(f"True Answer: {true_answer}")
        print(f"Predicted Answer: {predicted_answer}")
