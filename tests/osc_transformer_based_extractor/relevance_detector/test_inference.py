"""Test module for inference functions.

This module contains test cases for the inference functions in the
osc_transformer_based_extractor.inference module.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest
import torch

from osc_transformer_based_extractor.relevance_detector.inference import (
    get_batch_inference,
    run_full_inference,
)

# Define test data paths
model_path_valid = "valid_model"
model_path_invalid = "invalid_model"
tokenizer_path_valid = "valid_tokenizer"
tokenizer_path_invalid = "invalid_tokenizer"

# Create dummy model and tokenizer directories for testing
os.makedirs(model_path_valid, exist_ok=True)
os.makedirs(tokenizer_path_valid, exist_ok=True)


@patch(
    "osc_transformer_based_extractor.relevance_detector.inference.AutoModelForSequenceClassification.from_pretrained"
)
@patch(
    "osc_transformer_based_extractor.relevance_detector.inference.AutoTokenizer.from_pretrained"
)
def test_get_batch_inference():
    """Test the get_batch_inference function for inference correctness."""

    # Mock tokenizer and model
    tokenizer_mock = MagicMock()
    model_mock = MagicMock()

    # Configure the tokenizer mock for batch inputs
    tokenizer_mock.return_value = {
        "input_ids": torch.tensor([[101, 102], [101, 103]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
    }

    # Configure the model mock to return logits tensor
    model_output_mock = MagicMock()
    model_output_mock.logits = torch.tensor([[0.1, 0.9], [0.7, 0.3]])
    model_mock.return_value = model_output_mock

    # Dummy questions and contexts
    questions = ["What is the capital of France?", "What is the capital of Germany?"]
    contexts = ["Paris is the capital of France.", "Berlin is the capital of Germany."]

    # Test inference
    labels, positive_class_probs = get_batch_inference(
        questions, contexts, model_mock, tokenizer_mock, device="cpu", threshold=0.7
    )

    # Assertions
    assert isinstance(labels, np.ndarray) and labels.dtype == int
    assert (
        isinstance(positive_class_probs, np.ndarray)
        and positive_class_probs.dtype == float
    )

    # Check correct shapes and values
    assert labels.shape[0] == len(questions)
    assert positive_class_probs.shape[0] == len(questions)


@pytest.fixture
def sample_json_data():
    """Returns sample JSON data."""
    return {
        "dictionary": {
            "1": {
                "0": {
                    "pdf_name": "sample.pdf",
                    "unique_paragraph_id": "123",
                    "paragraph": "This is a sample paragraph.",
                }
            }
        }
    }


@pytest.fixture
def sample_kpi_mapping():
    """Returns a sample KPI mapping dataframe."""
    return pd.DataFrame(
        {"kpi_id": [1, 2], "question": ["What is the revenue?", "What is the profit?"]}
    )


@pytest.fixture
def sample_dataframe():
    """Returns a sample dataframe."""
    return pd.DataFrame(
        {
            "page": [1],
            "pdf_name": ["sample.pdf"],
            "unique_paragraph_id": ["123"],
            "paragraph": ["This is a sample paragraph."],
        }
    )


@pytest.fixture
def sample_merged_dataframe(sample_dataframe, sample_kpi_mapping):
    """Returns a merged dataframe."""
    return sample_dataframe.merge(sample_kpi_mapping, how="cross")


@patch(
    "builtins.open", new_callable=mock_open, read_data=json.dumps({"dictionary": {}})
)
@patch("pandas.read_csv")
@patch("os.listdir")
@patch("osc_transformer_based_extractor.relevance_detector.inference.get_inference")
@patch("pandas.DataFrame.to_excel")
def test_run_full_inference(
    mock_to_excel,
    mock_get_batch_inference,
    mock_listdir,
    mock_read_csv,
    mock_open,
    sample_json_data,
    sample_kpi_mapping,
    sample_merged_dataframe,
):
    """Test the run_full_inference function."""
    folder_path = "test_folder"
    kpi_mapping_path = "test_kpi.csv"
    output_path = "output_folder"
    model_path = "model_path"
    tokenizer_path = "tokenizer_path"
    batch_size = 2
    threshold = 0.5

    # Set up mock returns
    mock_read_csv.return_value = sample_kpi_mapping
    mock_listdir.return_value = ["test_file.json"]
    mock_open.return_value.read = json.dumps(sample_json_data)
    mock_get_batch_inference.return_value = (
        [1, 0],
        [0.95, 0.3],
    )  # mock labels and probabilities

    with patch("json.load", return_value=sample_json_data):
        with patch("pandas.DataFrame.merge", return_value=sample_merged_dataframe):
            with patch("pandas.DataFrame.to_excel", mock_to_excel):
                run_full_inference(
                    folder_path,
                    kpi_mapping_path,
                    output_path,
                    model_path,
                    tokenizer_path,
                    batch_size,
                    threshold,
                )

    # Assertions
    mock_read_csv.assert_called_once_with(kpi_mapping_path)
    mock_listdir.assert_called_once_with(folder_path)
    mock_open.assert_called_once_with(Path(folder_path) / "test_file.json", "r")

    # Check if batch inference was called with expected count
    assert (
        mock_get_batch_inference.call_count
        == (len(sample_merged_dataframe) // batch_size) + 1
    )
    assert mock_to_excel.call_count == 1
    output_file_path = Path(output_path) / "test_file.xlsx"
    mock_to_excel.assert_called_once_with(output_file_path, index=False)
