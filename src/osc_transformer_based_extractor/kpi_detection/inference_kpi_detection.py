"""
## Inference
"""

import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoConfig
from transformers.pipelines import QuestionAnsweringPipeline


def resolve_model_path(model_path: str):
    """
    Resolves whether the given `model_path` is a Hugging Face model name or a local system path.

    - If the `model_path` refers to a Hugging Face model (e.g., "bert-base-uncased"), the function will return the
      model name as a string.
    - If the `model_path` refers to a valid local system path, the function will convert it into a `Path` object.
    - If neither, the function raises a `ValueError`.
    """

    # Check if it's a local path
    if os.path.exists(model_path):
        return Path(model_path)

    # Check if it's a Hugging Face model name
    try:
        AutoConfig.from_pretrained(model_path)
        return model_path  # It's a Hugging Face model name, return as string
    except Exception:
        raise ValueError(
            f"{model_path} is neither a valid Hugging Face model nor a local file path."
        )


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


def get_batch_inference_kpi_detection(
    questions, contexts, question_answerer: QuestionAnsweringPipeline, batch_size
):
    """
    Perform batch inference using the question-answering pipeline.

    Args:
        questions (list): List of questions.
        contexts (list): List of contexts.
        question_answerer (QuestionAnsweringPipeline): The question-answering pipeline.
        batch_size (int): The batch size for inference.

    Returns:
        list of dict: List of dictionaries containing answers, scores, and positions.
    """
    results = question_answerer(
        questions, contexts, batch_size=batch_size
    )  # Adjust batch size as needed
    return results


def run_full_inference_kpi_detection(
    data_file_path: str,
    output_path: str,
    model_path: str,
    batch_size: int,
):
    """
    Runs full inference on a dataset of questions and contexts, and saves the results.

    Args:
        data_file_path (str): Path to the input CSV file containing the dataset.
            The dataset should have columns 'question' and 'context'.
        output_path (str): Path to the directory where the output Excel file will be saved.
        model_path (str): Path to the pre-trained model to be used for inference.
        batch_size (int): The batch size for inference.

    Returns:
        None: The function saves the resulting DataFrame to an Excel file and prints a success message.
    """
    data_file_path = str(Path(data_file_path))
    output_path = str(Path(output_path))
    model_path = resolve_model_path(model_path)

    data = pd.read_csv(data_file_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use NVIDIA GPU
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Use Apple Silicon GPU (MPS)
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")  # Fallback to CPU
        print("Using CPU")

    # Initialize the question-answering pipeline
    question_answerer = pipeline("question-answering", model=model_path, device=device)

    results = []

    # Process in batches
    for start_idx in tqdm(
        range(0, data.shape[0], batch_size), desc="Processing Batches"
    ):
        end_idx = min(start_idx + batch_size, data.shape[0])
        batch_questions = data["question"].iloc[start_idx:end_idx].tolist()
        batch_contexts = data["context"].iloc[start_idx:end_idx].tolist()

        # Perform batch inference
        batch_results = get_batch_inference_kpi_detection(
            questions=batch_questions,
            contexts=batch_contexts,
            question_answerer=question_answerer,
            batch_size=batch_size,
        )

        for result in batch_results:
            results.append(
                {
                    "predicted_answer": result["answer"],
                    "start": result["start"],
                    "end": result["end"],
                    "score": result["score"],
                }
            )

    df = pd.DataFrame(results)
    combined_df = pd.concat([data, df], axis=1)

    file_name = Path(output_path) / "output.xlsx"
    combined_df.to_excel(file_name, index=False)
    print(f"Successfully SAVED resulting file at {file_name}")
