"""This module contains utility functions for performing inference using pre-trained sequence classification models."""

# Module: inference
# Author: Tanishq-ids

import os
import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def validate_path_exists(path: str, which_path: str):
    """
    Validate if the given path exists.

    Args:
        path (str): The path to validate.

    Returns:
        ValueError: If the path does not exist.
    """
    if not os.path.exists(path):
        raise ValueError(f"{which_path}: {path} does not exist.")


def get_inference(
    question: str, context: str, model_path: str, tokenizer_path: str, threshold: float
):
    """
    Perform inference using a pre-trained sequence classification model.

    Parameters:
        question (str): The question for inference.
        context (str): The context to be analyzed.
        model_path (str): Path to the pre-trained model directory OR name on huggingface.
        tokenizer_path (str): Path to the tokenizer directory OR name on huggingface.
        threshold (float): The threshold for the inference score.

    Returns:
        int: Predicted label ID (0 or 1).
        float: class probability
    """
    model_path = str(Path(model_path))
    tokenizer_path = str(Path(tokenizer_path))

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Tokenize inputs
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    # Probability of the positive class (label 1)
    positive_class_prob = probabilities[0, 1].item()

    label = 1 if positive_class_prob >= threshold else 0

    return label, positive_class_prob


def run_full_inference(
    json_folder_path: str,
    kpi_mapping_path: str,
    output_path: str,
    model_path: str,
    tokenizer_path: str,
    threshold: float,
):
    """
    Perform inference on JSON files in a specified folder and save the results to Excel files.

    This function reads JSON files from a specified folder, processes the data, merges it with a KPI mapping,
    performs inference using a specified model and tokenizer, and saves the results to Excel files.

    Args:
        json_folder_path (str): Path to the folder containing JSON files to process.
        kpi_mapping_path (str): Path to the CSV file containing KPI mappings.
        output_path (str): Path to the folder where the output Excel files will be saved.
        model_path (str): Path to the model used for inference.
        tokenizer_path (str): Path to the tokenizer used for inference.
        threshold (float): Threshold value for the inference process.

    Returns:
        None

    Raises:
        Exception: If there is an error reading or processing the JSON files, or saving the Excel files.
    """
    kpi_mapping_path = str(Path(kpi_mapping_path))
    json_folder_path = str(Path(json_folder_path))
    output_path = str(Path(output_path))
    model_path = str(Path(model_path))
    tokenizer_path = str(Path(tokenizer_path))

    # Read the KPI mapping outside the loop
    kpi_mapping = pd.read_csv(kpi_mapping_path)
    kpi_mapping = kpi_mapping[["kpi_id", "question"]]

    for file_name in os.listdir(json_folder_path):
        if file_name.endswith(".json"):
            try:
                with open(Path(json_folder_path) / file_name, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue

            data_list = []

            # Iterate through the nested dictionary structure
            for page_idx in data:
                for paras in data[page_idx]:
                    pdf_name = data[page_idx][paras]["pdf_name"]
                    unique_paragraph_id = data[page_idx][paras]["unique_paragraph_id"]
                    paragraph = data[page_idx][paras]["paragraph"]

                    # Append a dictionary to the list
                    data_list.append(
                        {
                            "page": page_idx,
                            "pdf_name": pdf_name,
                            "unique_paragraph_id": unique_paragraph_id,
                            "paragraph": paragraph,
                        }
                    )

            df = pd.DataFrame(data_list)

            # Merge with KPI mapping using a cross join
            merged_df = df.merge(kpi_mapping, how="cross")

            labels = []
            probs = []

            # Iterate over the rows of the DataFrame and perform inference
            for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
                question = row["question"]
                context = row["paragraph"]
                label, prob = get_inference(
                    question=question,
                    context=context,
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    threshold=threshold,
                )
                labels.append(label)
                probs.append(prob)

            # Add results to the DataFrame
            merged_df["paragraph_relevance_flag"] = labels
            merged_df["paragraph_relevance_score(for_label=1)"] = probs

            excel_name = Path(file_name).with_suffix(".xlsx").name
            output_file_path = Path(output_path) / excel_name

            try:
                merged_df.to_excel(output_file_path, index=False)
                print(f"Successfully performed relevance for {file_name}")
                print(f"Successfully SAVED resulting file at {output_file_path}")
            except Exception as e:
                print(f"Error saving file {excel_name}: {e}")
