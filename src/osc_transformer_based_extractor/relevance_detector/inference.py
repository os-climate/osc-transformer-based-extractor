import os
import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


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


def resolve_model_path(model_path: str):
    """
    Resolves whether the given `model_path` is a Hugging Face model name or a local system path.
    """
    if os.path.exists(model_path):
        return Path(model_path)
    try:
        AutoConfig.from_pretrained(model_path)
        return model_path
    except Exception:
        raise ValueError(
            f"{model_path} is neither a valid Hugging Face model nor a local file path."
        )


def get_batch_inference(questions, contexts, model, tokenizer, device, threshold):
    """
    Perform batch inference using the model and tokenizer.
    """
    # Tokenize the batch of questions and contexts
    inputs = tokenizer(
        questions,
        contexts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )

    # Move inputs to the device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probabilities = torch.softmax(outputs.logits, dim=1)

    # Probability of the positive class (label 1)
    positive_class_probs = probabilities[:, 1].cpu().numpy()
    labels = (positive_class_probs >= threshold).astype(int)

    return labels, positive_class_probs


def run_full_inference(
    json_folder_path: str,
    kpi_mapping_path: str,
    output_path: str,
    model_path: str,
    tokenizer_path: str,
    batch_size: int,
    threshold: float,
):
    """
    Perform inference on JSON files in a specified folder and save the results to Excel files.
    """
    kpi_mapping_path = str(Path(kpi_mapping_path))
    json_folder_path = str(Path(json_folder_path))
    output_path = str(Path(output_path))

    # Resolve model and tokenizer paths
    model_path = resolve_model_path(model_path)
    tokenizer_path = resolve_model_path(tokenizer_path)

    # Load model and tokenizer once
    # Dynamically detect the device: CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Apple Silicon GPU (MPS)
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        print("Using CPU")

    print(f"Using device: {device}")  # Print device to confirm
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    model.eval()  # Set model to evaluation mode

    # Read the KPI mapping
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

            # Perform inference in batches
            for start_idx in tqdm(range(0, merged_df.shape[0], batch_size)):
                end_idx = min(start_idx + batch_size, merged_df.shape[0])
                batch_questions = merged_df["question"].iloc[start_idx:end_idx].tolist()
                batch_contexts = merged_df["paragraph"].iloc[start_idx:end_idx].tolist()

                batch_labels, batch_probs = get_batch_inference(
                    batch_questions, batch_contexts, model, tokenizer, device, threshold
                )

                labels.extend(batch_labels)
                probs.extend(batch_probs)

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
