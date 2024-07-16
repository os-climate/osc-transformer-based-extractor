"""This module contains utility functions for performing inference using pre-trained sequence classification models."""

# Module: inference
# Author: Tanishq-ids


import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def check_model_and_tokenizer_path(model_path, tokenizer_path):
    """
    Check if the model and tokenizer paths are valid.

    Args:
        model_path (str): Path to the model file.
        tokenizer_path (str): Path to the tokenizer file.

    Raises:
        ValueError: If the model or tokenizer path does not exist.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist.")

    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer path {tokenizer_path} does not exist.")


def check_question_context(question, context):
    """
    Check if the question and context are valid strings and not empty.

    Args:
        question (str): The question string.
        context (str): The context string.

    Raises:
        ValueError: If the question or context is not a string or is empty.
    """
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    if not isinstance(context, str):
        raise ValueError("Context must be a string.")

    if not question.strip():
        raise ValueError("Question is empty.")

    if not context.strip():
        raise ValueError("Context is empty.")


def get_inference(question: str, context: str, model_path: str, tokenizer_path: str) -> int:
    """
    Perform inference using a pre-trained sequence classification model.

    Parameters:
        question (str): The question for inference.
        context (str): The context to be analyzed.
        model_path (str): Path to the pre-trained model.
        tokenizer_path (str): Path to the tokenizer to the pre-trained model.

    Returns:
        int: Predicted label ID.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Tokenize inputs
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    predicted_label_id = torch.argmax(outputs.logits, dim=1).item()

    return predicted_label_id
