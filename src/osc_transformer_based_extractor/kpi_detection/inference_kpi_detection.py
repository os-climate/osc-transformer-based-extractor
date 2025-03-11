"""
## Inference
"""

import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.pipelines import QuestionAnsweringPipeline
import re


torch.random.manual_seed(0)


def extract_paragraph_number(text: str):
    """
    Extract the first number from a substring in the format 'Paragraph {num}'.

    Args:
        text (str): The input string.

    Returns:
        int or None: The extracted number, or None if no match is found.
    """
    # Regex pattern to match "Paragraph {num}"
    pattern = r"Paragraph (\d+)"

    # Search for the pattern
    match = re.search(pattern, text)

    if match:
        return int(match.group(1))  # Extract and convert the number
    return None


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
    # Combine questions and contexts into a list of dictionaries
    inputs = [{"question": q, "context": c} for q, c in zip(questions, contexts)]

    results = []
    # Process in batches
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]  # Get batch
        batch_results = question_answerer(batch)  # Perform inference on the batch
        results.append(batch_results)

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
        data_file_path (str): Path to the input CSV/Excel file containing the dataset.
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

    if data_file_path.endswith(".csv"):
        data = pd.read_csv(data_file_path)
    else:
        data = pd.read_excel(data_file_path)

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
            print()
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
    if "Unnamed: 0" in combined_df.columns:
        combined_df.drop(columns=["Unnamed: 0"], inplace=True)
    combined_df.sort_values(
        by=["pdf_name", "kpi_id", "score"], inplace=True, ascending=[True, True, False]
    )
    combined_df = combined_df[
        [
            "pdf_name",
            "question",
            "predicted_answer",
            "score",
            "page",
            "context",
            "kpi_id",
            "unique_paragraph_id",
            "paragraph_relevance_score(for_label=1)",
            "paragraph_relevance_flag",
            "start",
            "end",
        ]
    ]

    verifier_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct", torch_dtype="auto", trust_remote_code=True
    ).to(device)

    verifier_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct"
    )
    verified_df = llm_2fv(combined_df, verifier_model, verifier_tokenizer, device)

    file_name = Path(output_path) / "output.xlsx"
    verified_df.to_excel(file_name, index=False)
    print(f"Successfully SAVED resulting file at {file_name}")


def llm_2fv(
    df: pd.DataFrame,
    verifier_model: PreTrainedModel,
    verifier_tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> pd.DataFrame:
    """
    Processes a DataFrame of PDF-extracted questions and answers, uses an LLM-based verifier
    to select the most relevant paragraph-answer pair, and returns the most relevant results.

    Args:
        df (pd.DataFrame):
            A DataFrame containing columns:
            - 'pdf_name': Name of the PDF document.
            - 'kpi_id': Unique identifier for the KPI.
            - 'paragraph_relevance_score(for_label=1)': Relevance score for the paragraph.
            - 'question': Question related to the KPI.
            - 'context': Paragraph context from the PDF.
            - 'predicted_answer': Answer predicted for the context.
        verifier_model (PreTrainedModel):
            The language model used to evaluate paragraph-answer pairs.
        verifier_tokenizer (PreTrainedTokenizer):
            Tokenizer associated with the language model.
        device (torch.device):
            Device to run the model on, e.g., 'cpu' or 'cuda'.

    Returns:
        pd.DataFrame: A DataFrame containing the most relevant paragraph-answer pair for each PDF and KPI.
    """
    result_rows = []

    # Group by 'pdf_name' and 'kpi_id'
    grouped = df.groupby(["pdf_name", "kpi_id"])

    for (pdf_name, kpi_id), group in grouped:
        group = group.sort_values(
            by=["paragraph_relevance_score(for_label=1)"], ascending=False
        ).head(4)
        question = group["question"].iloc[0]

        pairs = [
            f"**Paragraph {i + 1}**: {p}\n   **Answer {i + 1}**: {a}\n"
            for i, (p, a) in enumerate(zip(group["context"], group["predicted_answer"]))
        ]
        combined_pairs = "\n\n".join(pairs)

        content = (
            f"### Instruction:\n"
            f"You are tasked with analyzing a question extracted from a PDF, along with multiple paragraph-answer pairs. "
            f"Your task is to identify the **most relevant paragraph-answer pair** based on the given information and explain your reasoning.\n\n"
            f"### Input:\n"
            f"**Question**: {question}\n\n"
            f"**Paragraph-Answer Pairs**:\n\n"
            f"{combined_pairs}\n\n"
            f"### Task:\n"
            f"1. Identify which **paragraph-answer pair** is the most relevant.\n"
            f"2. Give me the page number.\n\n"
            f"3. Provide a clear and concise explanation for your choice, considering the content.\n\n"
            f"### Response:\n"
        )

        inputs = verifier_tokenizer(content, return_tensors="pt").to(device)
        outputs = verifier_model.generate(
            **inputs, max_new_tokens=500, temperature=0.0, do_sample=False
        )
        output_text = verifier_tokenizer.decode(outputs[0], skip_special_tokens=True)

        num = extract_paragraph_number(output_text)
        result_rows.append(group.iloc[num - 1])

    return pd.DataFrame(result_rows).reset_index(drop=True)
